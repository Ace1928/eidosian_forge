from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CheckUpgradeRequest(_messages.Message):
    """Request to check whether image upgrade will succeed.

  Fields:
    imageVersion: The version of the software running in the environment. This
      encapsulates both the version of Cloud Composer functionality and the
      version of Apache Airflow. It must match the regular expression `compose
      r-([0-9]+(\\.[0-9]+\\.[0-9]+(-preview\\.[0-9]+)?)?|latest)-airflow-([0-
      9]+(\\.[0-9]+(\\.[0-9]+)?)?)`. When used as input, the server also checks
      if the provided version is supported and denies the request for an
      unsupported version. The Cloud Composer portion of the image version is
      a full [semantic version](https://semver.org), or an alias in the form
      of major version number or `latest`. When an alias is provided, the
      server replaces it with the current Cloud Composer version that
      satisfies the alias. The Apache Airflow portion of the image version is
      a full semantic version that points to one of the supported Apache
      Airflow versions, or an alias in the form of only major or major.minor
      versions specified. When an alias is provided, the server replaces it
      with the latest Apache Airflow version that satisfies the alias and is
      supported in the given Cloud Composer version. In all cases, the
      resolved image version is stored in the same field. See also [version
      list](/composer/docs/concepts/versioning/composer-versions) and
      [versioning overview](/composer/docs/concepts/versioning/composer-
      versioning-overview).
  """
    imageVersion = _messages.StringField(1)