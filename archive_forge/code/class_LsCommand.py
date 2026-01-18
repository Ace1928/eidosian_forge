from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
import six
from gslib.cloud_api import NotFoundException
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.storage_url import ContainsWildcard
from gslib.storage_url import StorageUrlFromString
from gslib.utils.constants import NO_MAX
from gslib.utils.constants import S3_DELETE_MARKER_GUID
from gslib.utils.constants import UTF8
from gslib.utils.ls_helper import ENCRYPTED_FIELDS
from gslib.utils.ls_helper import LsHelper
from gslib.utils.ls_helper import PrintFullInfoAboutObject
from gslib.utils.ls_helper import UNENCRYPTED_FULL_LISTING_FIELDS
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.text_util import InsistAscii
from gslib.utils import text_util
from gslib.utils.translation_helper import AclTranslation
from gslib.utils.translation_helper import LabelTranslation
from gslib.utils.unit_util import MakeHumanReadable
class LsCommand(Command):
    """Implementation of gsutil ls command."""
    command_spec = Command.CreateCommandSpec('ls', command_name_aliases=['dir', 'list'], usage_synopsis=_SYNOPSIS, min_args=0, max_args=NO_MAX, supported_sub_args='aebdlLhp:rR', file_url_ok=False, provider_url_ok=True, urls_start_arg=0, gs_api_support=[ApiSelector.XML, ApiSelector.JSON], gs_default_api=ApiSelector.JSON, argparse_arguments=[CommandArgument.MakeZeroOrMoreCloudURLsArgument()])
    help_spec = Command.HelpSpec(help_name='ls', help_name_aliases=['dir', 'list'], help_type='command_help', help_one_line_summary='List providers, buckets, or objects', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={})
    gcloud_storage_map = GcloudStorageMap(gcloud_command=['storage', 'ls', '--fetch-encrypted-object-hashes'], flag_map={'-r': GcloudStorageFlag('-r'), '-R': GcloudStorageFlag('-r'), '-l': GcloudStorageFlag('-l'), '-L': GcloudStorageFlag('-L'), '-b': GcloudStorageFlag('-b'), '-e': GcloudStorageFlag('-e'), '-a': GcloudStorageFlag('-a'), '-h': GcloudStorageFlag('--readable-sizes'), '-p': GcloudStorageFlag('--project')})

    def _PrintBucketInfo(self, bucket_blr, listing_style):
        """Print listing info for given bucket.

    Args:
      bucket_blr: BucketListingReference for the bucket being listed
      listing_style: ListingStyle enum describing type of output desired.

    Returns:
      Tuple (total objects, total bytes) in the bucket.
    """
        if listing_style == ListingStyle.SHORT or listing_style == ListingStyle.LONG:
            text_util.print_to_fd(bucket_blr)
            return
        bucket = bucket_blr.root_object
        location_constraint = bucket.location
        storage_class = bucket.storageClass
        fields = {'bucket': bucket_blr.url_string, 'storage_class': storage_class, 'location_constraint': location_constraint, 'acl': AclTranslation.JsonFromMessage(bucket.acl), 'default_acl': AclTranslation.JsonFromMessage(bucket.defaultObjectAcl), 'versioning': bucket.versioning and bucket.versioning.enabled, 'website_config': 'Present' if bucket.website else 'None', 'logging_config': 'Present' if bucket.logging else 'None', 'cors_config': 'Present' if bucket.cors else 'None', 'lifecycle_config': 'Present' if bucket.lifecycle else 'None', 'requester_pays': bucket.billing and bucket.billing.requesterPays}
        if bucket.retentionPolicy:
            fields['retention_policy'] = 'Present'
        if bucket.labels:
            fields['labels'] = LabelTranslation.JsonFromMessage(bucket.labels, pretty_print=True)
        else:
            fields['labels'] = 'None'
        if bucket.encryption and bucket.encryption.defaultKmsKeyName:
            fields['default_kms_key'] = bucket.encryption.defaultKmsKeyName
        else:
            fields['default_kms_key'] = 'None'
        fields['encryption_config'] = 'Present' if bucket.encryption else 'None'
        if bucket.autoclass and bucket.autoclass.enabled:
            fields['autoclass_enabled_date'] = bucket.autoclass.toggleTime.strftime('%a, %d %b %Y')
        if bucket.locationType:
            fields['location_type'] = bucket.locationType
        if bucket.customPlacementConfig:
            fields['custom_placement_locations'] = bucket.customPlacementConfig.dataLocations
        if bucket.metageneration:
            fields['metageneration'] = bucket.metageneration
        if bucket.timeCreated:
            fields['time_created'] = bucket.timeCreated.strftime('%a, %d %b %Y %H:%M:%S GMT')
        if bucket.updated:
            fields['updated'] = bucket.updated.strftime('%a, %d %b %Y %H:%M:%S GMT')
        if bucket.defaultEventBasedHold:
            fields['default_eventbased_hold'] = bucket.defaultEventBasedHold
        if bucket.iamConfiguration:
            if bucket.iamConfiguration.bucketPolicyOnly:
                enabled = bucket.iamConfiguration.bucketPolicyOnly.enabled
                fields['bucket_policy_only_enabled'] = enabled
            if bucket.iamConfiguration.publicAccessPrevention:
                fields['public_access_prevention'] = bucket.iamConfiguration.publicAccessPrevention
        if bucket.rpo:
            fields['rpo'] = bucket.rpo
        if bucket.satisfiesPZS:
            fields['satisfies_pzs'] = bucket.satisfiesPZS
        for key in fields:
            previous_value = fields[key]
            if not isinstance(previous_value, six.string_types) or '\n' not in previous_value:
                continue
            new_value = previous_value.replace('\n', '\n\t  ')
            if not new_value.startswith('\n'):
                new_value = '\n\t  ' + new_value
            fields[key] = new_value
        autoclass_line = ''
        location_type_line = ''
        custom_placement_locations_line = ''
        metageneration_line = ''
        time_created_line = ''
        time_updated_line = ''
        default_eventbased_hold_line = ''
        retention_policy_line = ''
        bucket_policy_only_enabled_line = ''
        public_access_prevention_line = ''
        rpo_line = ''
        satisifies_pzs_line = ''
        if 'autoclass_enabled_date' in fields:
            autoclass_line = '\tAutoclass:\t\t\tEnabled on {autoclass_enabled_date}\n'
        if 'location_type' in fields:
            location_type_line = '\tLocation type:\t\t\t{location_type}\n'
        if 'custom_placement_locations' in fields:
            custom_placement_locations_line = '\tPlacement locations:\t\t{custom_placement_locations}\n'
        if 'metageneration' in fields:
            metageneration_line = '\tMetageneration:\t\t\t{metageneration}\n'
        if 'time_created' in fields:
            time_created_line = '\tTime created:\t\t\t{time_created}\n'
        if 'updated' in fields:
            time_updated_line = '\tTime updated:\t\t\t{updated}\n'
        if 'default_eventbased_hold' in fields:
            default_eventbased_hold_line = '\tDefault Event-Based Hold:\t{default_eventbased_hold}\n'
        if 'retention_policy' in fields:
            retention_policy_line = '\tRetention Policy:\t\t{retention_policy}\n'
        if 'bucket_policy_only_enabled' in fields:
            bucket_policy_only_enabled_line = '\tBucket Policy Only enabled:\t{bucket_policy_only_enabled}\n'
        if 'public_access_prevention' in fields:
            public_access_prevention_line = '\tPublic access prevention:\t{public_access_prevention}\n'
        if 'rpo' in fields:
            rpo_line = '\tRPO:\t\t\t\t{rpo}\n'
        if 'satisfies_pzs' in fields:
            satisifies_pzs_line = '\tSatisfies PZS:\t\t\t{satisfies_pzs}\n'
        text_util.print_to_fd(('{bucket} :\n\tStorage class:\t\t\t{storage_class}\n' + location_type_line + '\tLocation constraint:\t\t{location_constraint}\n' + custom_placement_locations_line + '\tVersioning enabled:\t\t{versioning}\n\tLogging configuration:\t\t{logging_config}\n\tWebsite configuration:\t\t{website_config}\n\tCORS configuration: \t\t{cors_config}\n\tLifecycle configuration:\t{lifecycle_config}\n\tRequester Pays enabled:\t\t{requester_pays}\n' + retention_policy_line + default_eventbased_hold_line + '\tLabels:\t\t\t\t{labels}\n' + '\tDefault KMS key:\t\t{default_kms_key}\n' + time_created_line + time_updated_line + metageneration_line + bucket_policy_only_enabled_line + autoclass_line + public_access_prevention_line + rpo_line + satisifies_pzs_line + '\tACL:\t\t\t\t{acl}\n\tDefault ACL:\t\t\t{default_acl}').format(**fields))
        if bucket_blr.storage_url.scheme == 's3':
            text_util.print_to_fd('Note: this is an S3 bucket so configuration values may be blank. To retrieve bucket configuration values, use individual configuration commands such as gsutil acl get <bucket>.')

    def _PrintLongListing(self, bucket_listing_ref):
        """Prints an object with ListingStyle.LONG."""
        obj = bucket_listing_ref.root_object
        url_str = bucket_listing_ref.url_string
        if obj.metadata and S3_DELETE_MARKER_GUID in obj.metadata.additionalProperties:
            size_string = '0'
            num_bytes = 0
            num_objs = 0
            url_str += '<DeleteMarker>'
        else:
            size_string = MakeHumanReadable(obj.size) if self.human_readable else str(obj.size)
            num_bytes = obj.size
            num_objs = 1
        timestamp = JSON_TIMESTAMP_RE.sub('\\1T\\2Z', str(obj.timeCreated))
        printstr = '%(size)10s  %(timestamp)s  %(url)s'
        encoded_etag = None
        encoded_metagen = None
        if self.all_versions:
            printstr += '  metageneration=%(metageneration)s'
            encoded_metagen = str(obj.metageneration)
        if self.include_etag:
            printstr += '  etag=%(etag)s'
            encoded_etag = obj.etag
        format_args = {'size': size_string, 'timestamp': timestamp, 'url': url_str, 'metageneration': encoded_metagen, 'etag': encoded_etag}
        text_util.print_to_fd(printstr % format_args)
        return (num_objs, num_bytes)

    def RunCommand(self):
        """Command entry point for the ls command."""
        got_nomatch_errors = False
        got_bucket_nomatch_errors = False
        listing_style = ListingStyle.SHORT
        get_bucket_info = False
        self.recursion_requested = False
        self.all_versions = False
        self.include_etag = False
        self.human_readable = False
        self.list_subdir_contents = True
        if self.sub_opts:
            for o, a in self.sub_opts:
                if o == '-a':
                    self.all_versions = True
                elif o == '-e':
                    self.include_etag = True
                elif o == '-b':
                    get_bucket_info = True
                elif o == '-h':
                    self.human_readable = True
                elif o == '-l':
                    listing_style = ListingStyle.LONG
                elif o == '-L':
                    listing_style = ListingStyle.LONG_LONG
                elif o == '-p':
                    InsistAscii(a, 'Invalid non-ASCII character found in project ID')
                    self.project_id = a
                elif o == '-r' or o == '-R':
                    self.recursion_requested = True
                elif o == '-d':
                    self.list_subdir_contents = False
        if not self.args:
            self.args = ['gs://']
        total_objs = 0
        total_bytes = 0

        def MaybePrintBucketHeader(blr):
            if len(self.args) > 1:
                text_util.print_to_fd('%s:' % six.ensure_text(blr.url_string))
        print_bucket_header = MaybePrintBucketHeader
        for url_str in self.args:
            storage_url = StorageUrlFromString(url_str)
            if storage_url.IsFileUrl():
                raise CommandException('Only cloud URLs are supported for %s' % self.command_name)
            bucket_fields = None
            if listing_style == ListingStyle.SHORT or listing_style == ListingStyle.LONG:
                bucket_fields = ['id']
            elif listing_style == ListingStyle.LONG_LONG:
                bucket_fields = ['acl', 'autoclass', 'billing', 'cors', 'customPlacementConfig', 'defaultObjectAcl', 'encryption', 'iamConfiguration', 'labels', 'location', 'locationType', 'logging', 'lifecycle', 'metageneration', 'retentionPolicy', 'defaultEventBasedHold', 'rpo', 'satisfiesPZS', 'storageClass', 'timeCreated', 'updated', 'versioning', 'website']
            if storage_url.IsProvider():
                for blr in self.WildcardIterator('%s://*' % storage_url.scheme).IterBuckets(bucket_fields=bucket_fields):
                    self._PrintBucketInfo(blr, listing_style)
            elif storage_url.IsBucket() and get_bucket_info:
                total_buckets = 0
                for blr in self.WildcardIterator(url_str).IterBuckets(bucket_fields=bucket_fields):
                    if not ContainsWildcard(url_str) and (not blr.root_object):
                        self.gsutil_api.GetBucket(blr.storage_url.bucket_name, fields=['id'], provider=storage_url.scheme)
                    self._PrintBucketInfo(blr, listing_style)
                    total_buckets += 1
                if not ContainsWildcard(url_str) and (not total_buckets):
                    got_bucket_nomatch_errors = True
            else:

                def _PrintPrefixLong(blr):
                    text_util.print_to_fd('%-33s%s' % ('', six.ensure_text(blr.url_string)))
                if listing_style == ListingStyle.SHORT:
                    listing_helper = LsHelper(self.WildcardIterator, self.logger, all_versions=self.all_versions, print_bucket_header_func=print_bucket_header, should_recurse=self.recursion_requested, list_subdir_contents=self.list_subdir_contents)
                elif listing_style == ListingStyle.LONG:
                    bucket_listing_fields = ['name', 'size', 'timeCreated', 'updated']
                    if self.all_versions:
                        bucket_listing_fields.extend(['generation', 'metageneration'])
                    if self.include_etag:
                        bucket_listing_fields.append('etag')
                    listing_helper = LsHelper(self.WildcardIterator, self.logger, print_object_func=self._PrintLongListing, print_dir_func=_PrintPrefixLong, print_bucket_header_func=print_bucket_header, all_versions=self.all_versions, should_recurse=self.recursion_requested, fields=bucket_listing_fields, list_subdir_contents=self.list_subdir_contents)
                elif listing_style == ListingStyle.LONG_LONG:
                    bucket_listing_fields = UNENCRYPTED_FULL_LISTING_FIELDS + ENCRYPTED_FIELDS
                    listing_helper = LsHelper(self.WildcardIterator, self.logger, print_object_func=PrintFullInfoAboutObject, print_dir_func=_PrintPrefixLong, print_bucket_header_func=print_bucket_header, all_versions=self.all_versions, should_recurse=self.recursion_requested, fields=bucket_listing_fields, list_subdir_contents=self.list_subdir_contents)
                else:
                    raise CommandException('Unknown listing style: %s' % listing_style)
                exp_dirs, exp_objs, exp_bytes = listing_helper.ExpandUrlAndPrint(storage_url)
                if storage_url.IsObject() and exp_objs == 0 and (exp_dirs == 0):
                    got_nomatch_errors = True
                total_bytes += exp_bytes
                total_objs += exp_objs
        if total_objs and listing_style != ListingStyle.SHORT:
            text_util.print_to_fd('TOTAL: %d objects, %d bytes (%s)' % (total_objs, total_bytes, MakeHumanReadable(float(total_bytes))))
        if got_nomatch_errors:
            raise CommandException('One or more URLs matched no objects.')
        if got_bucket_nomatch_errors:
            raise NotFoundException('One or more bucket URLs matched no buckets.')
        return 0