from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
import textwrap
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import errors as api_errors
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.storage import errors as command_errors
from googlecloudsdk.command_lib.storage import sign_url_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import iso_duration
from googlecloudsdk.core.util import times
class SignUrl(base.Command):
    """Generate a URL with embedded authentication that can be used by anyone."""
    detailed_help = {'DESCRIPTION': "\n      *{command}* will generate a signed URL that embeds authentication data so\n      the URL can be used by someone who does not have a Google account. Use the\n      global ``--impersonate-service-account'' flag to specify the service\n      account that will be used to sign the specified URL or authenticate with\n      a service account directly. Otherwise, a service account key is required.\n      Please see the [Signed URLs documentation](https://cloud.google.com/storage/docs/access-control/signed-urls)\n      for background about signed URLs.\n\n      Note, `{command}` does not support operations on sub-directories. For\n      example, unless you have an object named `some-directory/` stored inside\n      the bucket `some-bucket`, the following command returns an error:\n      `{command} gs://some-bucket/some-directory/`.\n      ", 'EXAMPLES': '\n      To create a signed url for downloading an object valid for 10 minutes with\n      the credentials of an impersonated service account:\n\n        $ {command} gs://my-bucket/file.txt --duration=10m --impersonate-service-account=sa@my-project.iam.gserviceaccount.com\n\n      To create a signed url that will bill to my-billing-project when already\n      authenticated as a service account:\n\n        $ {command} gs://my-bucket/file.txt --query-params=userProject=my-billing-project\n\n      To create a signed url, valid for one hour, for uploading a plain text\n      file via HTTP PUT:\n\n        $ {command} gs://my-bucket/file.txt --http-verb=PUT --duration=1h --headers=content-type=text/plain --impersonate-service-account=sa@my-project.iam.gserviceaccount.com\n\n      To create a signed URL that initiates a resumable upload for a plain text\n      file using a private key file:\n\n        $ {command} gs://my-bucket/file.txt --http-verb=POST --headers=x-goog-resumable=start,content-type=text/plain --private-key-file=key.json\n      '}

    @staticmethod
    def Args(parser):
        parser.add_argument('url', nargs='+', help='The URLs to be signed. May contain wildcards.')
        parser.add_argument('-d', '--duration', default=3600, type=arg_parsers.Duration(upper_bound='7d'), help=textwrap.dedent("            Specifies the duration that the signed url should be valid for,\n            default duration is 1 hour. For example 10s for 10 seconds.\n            See $ gcloud topic datetimes for information on duration formats.\n\n            The max duration allowed is 12 hours. This limitation exists because\n            the system-managed key used to sign the URL may not remain valid\n            after 12 hours.\n\n            Alternatively, the max duration allowed is 7 days when signing with\n            either the ``--private-key-file'' flag or an account that authorized\n            with ``gcloud auth activate-service-account''."))
        parser.add_argument('--headers', action=arg_parsers.UpdateAction, default={}, metavar='KEY=VALUE', type=arg_parsers.ArgDict(), help=textwrap.dedent("            Specifies the headers to be used in the signed request.\n            Possible headers are listed in the XML API's documentation:\n            https://cloud.google.com/storage/docs/xml-api/reference-headers#headers\n            "))
        parser.add_argument('-m', '--http-verb', default='GET', help=textwrap.dedent("            Specifies the HTTP verb to be authorized for use with the signed\n            URL, default is GET. When using a signed URL to start\n            a resumable upload session, you will need to specify the\n            ``x-goog-resumable:start'' header in the request or else signature\n            validation will fail."))
        parser.add_argument('--private-key-file', help=textwrap.dedent("            The service account private key used to generate the cryptographic\n            signature for the generated URL. Must be in PKCS12 or JSON format.\n            If encrypted, will prompt for the passphrase used to protect the\n            private key file (default ``notasecret'').\n\n            Note: Service account keys are a security risk if not managed\n            correctly. Review [best practices for managing service account keys](https://cloud.google.com/iam/docs/best-practices-for-managing-service-account-keys)\n            before using this option."))
        parser.add_argument('-p', '--private-key-password', help='Specifies the PRIVATE_KEY_FILE password instead of prompting.')
        parser.add_argument('--query-params', action=arg_parsers.UpdateAction, default={}, metavar='KEY=VALUE', type=arg_parsers.ArgDict(), help=textwrap.dedent("            Specifies the query parameters to be used in the signed request.\n            Possible query parameters are listed in the XML API's documentation:\n            https://cloud.google.com/storage/docs/xml-api/reference-headers#query\n            "))
        parser.add_argument('-r', '--region', help=textwrap.dedent("            Specifies the region in which the resources for which you are\n            creating signed URLs are stored.\n\n            Default value is ``auto'' which will cause {command} to fetch the\n            region for the resource. When auto-detecting the region, the current\n            user's credentials, not the credentials from PRIVATE_KEY_FILE,\n            are used to fetch the bucket's metadata."))

    def Run(self, args):
        key = None
        delegates = None
        creds = c_store.Load(prevent_refresh=True, use_google_auth=False)
        delegate_chain = args.impersonate_service_account or properties.VALUES.auth.impersonate_service_account.Get()
        if args.private_key_file:
            try:
                client_id, key = sign_url_util.get_signing_information_from_file(args.private_key_file, args.private_key_password)
            except ModuleNotFoundError as error:
                if 'OpenSSL' in str(error):
                    raise command_errors.Error(_INSTALL_PY_OPEN_SSL_MESSAGE)
                raise
        elif delegate_chain:
            impersonated_account, delegates = c_store.ParseImpersonationAccounts(delegate_chain)
            client_id = impersonated_account
        elif c_creds.CredentialType.FromCredentials(creds) == c_creds.CredentialType.GCE:
            client_id = properties.VALUES.core.account.Get()
        elif c_creds.IsServiceAccountCredentials(creds):
            try:
                client_id, key = sign_url_util.get_signing_information_from_json(c_creds.ToJson(creds))
            except ModuleNotFoundError as error:
                if 'OpenSSL' in str(error):
                    raise command_errors.Error(_INSTALL_PY_OPEN_SSL_MESSAGE)
                raise
        else:
            raise command_errors.Error(_PROVIDE_SERVICE_ACCOUNT_MESSAGE)
        host = properties.VALUES.storage.gs_xml_endpoint_url.Get()
        has_provider_url = any((storage_url.storage_url_from_string(url_string).is_provider() for url_string in args.url))
        if has_provider_url:
            raise command_errors.Error('The sign-url command does not support provider-only URLs.')
        for url_string in args.url:
            url = storage_url.storage_url_from_string(url_string)
            if wildcard_iterator.contains_wildcard(url_string):
                resources = wildcard_iterator.get_wildcard_iterator(url_string)
            else:
                resources = [resource_reference.UnknownResource(url)]
            for resource in resources:
                if resource.storage_url.is_bucket():
                    path = '/{}'.format(resource.storage_url.bucket_name)
                else:
                    path = '/{}/{}'.format(resource.storage_url.bucket_name, resource.storage_url.object_name)
                parameters = dict(args.query_params)
                if url.generation:
                    parameters['generation'] = url.generation
                region = _get_region(args, resource)
                signed_url = sign_url_util.get_signed_url(client_id=client_id, duration=args.duration, headers=args.headers, host=host, key=key, verb=args.http_verb, parameters=parameters, path=path, region=region, delegates=delegates)
                expiration_time = times.GetDateTimePlusDuration(times.Now(tzinfo=times.UTC), iso_duration.Duration(seconds=args.duration))
                yield {'resource': str(resource), 'http_verb': args.http_verb, 'expiration': times.FormatDateTime(expiration_time, fmt='%Y-%m-%d %H:%M:%S'), 'signed_url': signed_url}
                sign_url_util.probe_access_to_resource(client_id=client_id, host=host, key=key, path=path, region=region, requested_headers=args.headers, requested_http_verb=args.http_verb, requested_parameters=parameters, requested_resource=resource)