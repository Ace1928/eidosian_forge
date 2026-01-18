from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.command_lib.iam import iam_util
def GenerateContentForCreateRequest(args):
    """Creates Content for Message Create Requests."""
    module = dataplex_api.GetMessageModule()
    content = module.GoogleCloudDataplexV1Content(dataText=args.data_text, description=args.description, labels=dataplex_api.CreateLabels(module.GoogleCloudDataplexV1Content, args), path=args.path)
    if args.kernel_type:
        content.notebook = GenerateNotebook(args)
    if args.query_engine:
        content.sqlScript = GenerateSqlScript(args)
    return content