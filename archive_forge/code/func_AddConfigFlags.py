from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
import six
def AddConfigFlags(parser):
    """Add config flags."""
    build_config = parser.add_mutually_exclusive_group()
    build_config.add_argument('--tag', '-t', help='The tag to use with a "docker build" image creation. Cloud Build will run a remote "docker build -t $TAG .", where $TAG is the tag provided by this flag. The tag must be in the *gcr.io* or *pkg.dev* namespace. Specify a tag if you want Cloud Build to build using a Dockerfile instead of a build config file. If you specify a tag in this command, your source must include a Dockerfile. For instructions on building using a Dockerfile see https://cloud.google.com/cloud-build/docs/quickstart-build.')
    build_config.add_argument('--config', default='cloudbuild.yaml', help='The YAML or JSON file to use as the build configuration file.')
    build_config.add_argument('--pack', type=arg_parsers.ArgDict(spec={'image': str, 'builder': str, 'env': str}), action='append', help='Uses CNCF [buildpack](https://buildpacks.io/) to create image.  The "image" key/value must be provided.  The image name must be in the *gcr.io* or *pkg.dev* namespace. By default ```gcr.io/buildpacks/builder``` will be used. To specify your own builder image use the optional "builder" key/value argument.  To pass environment variables to the builder use the optional "env" key/value argument where value is a list of key values using [escaping](https://cloud.google.com/sdk/gcloud/reference/topic/escaping) if necessary.')