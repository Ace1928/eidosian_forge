from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.command_lib.iam import iam_util
def GenerateSqlScript(args):
    """Creates SQL Script field for Content Message Create/Update Requests."""
    module = dataplex_api.GetMessageModule()
    query_engine_field = module.GoogleCloudDataplexV1ContentSqlScript
    sql_script = module.GoogleCloudDataplexV1ContentSqlScript()
    if args.query_engine:
        sql_script.engine = query_engine_field.EngineValueValuesEnum(args.query_engine)
    return sql_script