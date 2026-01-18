import io
import json
import os
import six
from google.auth import _default
from google.auth import environment_vars
from google.auth import exceptions
Gets the default credentials for the current environment.

    `Application Default Credentials`_ provides an easy way to obtain
    credentials to call Google APIs for server-to-server or local applications.
    This function acquires credentials from the environment in the following
    order:

    1. If the environment variable ``GOOGLE_APPLICATION_CREDENTIALS`` is set
       to the path of a valid service account JSON private key file, then it is
       loaded and returned. The project ID returned is the project ID defined
       in the service account file if available (some older files do not
       contain project ID information).
    2. If the `Google Cloud SDK`_ is installed and has application default
       credentials set they are loaded and returned.

       To enable application default credentials with the Cloud SDK run::

            gcloud auth application-default login

       If the Cloud SDK has an active project, the project ID is returned. The
       active project can be set using::

            gcloud config set project

    3. If the application is running in the `App Engine standard environment`_
       (first generation) then the credentials and project ID from the
       `App Identity Service`_ are used.
    4. If the application is running in `Compute Engine`_ or `Cloud Run`_ or
       the `App Engine flexible environment`_ or the `App Engine standard
       environment`_ (second generation) then the credentials and project ID
       are obtained from the `Metadata Service`_.
    5. If no credentials are found,
       :class:`~google.auth.exceptions.DefaultCredentialsError` will be raised.

    .. _Application Default Credentials: https://developers.google.com            /identity/protocols/application-default-credentials
    .. _Google Cloud SDK: https://cloud.google.com/sdk
    .. _App Engine standard environment: https://cloud.google.com/appengine
    .. _App Identity Service: https://cloud.google.com/appengine/docs/python            /appidentity/
    .. _Compute Engine: https://cloud.google.com/compute
    .. _App Engine flexible environment: https://cloud.google.com            /appengine/flexible
    .. _Metadata Service: https://cloud.google.com/compute/docs            /storing-retrieving-metadata
    .. _Cloud Run: https://cloud.google.com/run

    Example::

        import google.auth

        credentials, project_id = google.auth.default()

    Args:
        scopes (Sequence[str]): The list of scopes for the credentials. If
            specified, the credentials will automatically be scoped if
            necessary.
        request (google.auth.transport.Request): An object used to make
            HTTP requests. This is used to detect whether the application
            is running on Compute Engine. If not specified, then it will
            use the standard library http client to make requests.
        quota_project_id (Optional[str]):  The project ID used for
            quota and billing.
    Returns:
        Tuple[~google.auth.credentials.Credentials, Optional[str]]:
            the current environment's credentials and project ID. Project ID
            may be None, which indicates that the Project ID could not be
            ascertained from the environment.

    Raises:
        ~google.auth.exceptions.DefaultCredentialsError:
            If no credentials were found, or if the credentials found were
            invalid.
    