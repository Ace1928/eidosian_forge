from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddTargetGcsUriGroup(parser):
    """Register flags for Target GCS URI."""
    target_group = parser.add_group(mutex=True, required=True, help='Export Destination - The path and storage where the export will be stored.')
    target_group.add_argument('--target-gcs-uri', metavar='TARGET_GCS_URI', help='The path to the folder in Google Cloud Storage where the export will be stored. The URI is in the form `gs://bucketName/folderName`. The Looker Service Agent should have the role Storage Object Creator.')