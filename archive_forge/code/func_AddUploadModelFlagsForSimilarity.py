from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
import textwrap
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.command_lib.iam import iam_util as core_iam_util
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def AddUploadModelFlagsForSimilarity(parser):
    """Adds flags for example-based explanation for UploadModel.

  Args:
    parser: the parser for the command.
  """
    base.Argument('--uris', metavar='URIS', type=arg_parsers.ArgList(), help='Cloud Storage bucket paths where training data is stored. Should be used only\nwhen the explanation method is `examples`.\n').AddToParser(parser)
    parser.add_argument('--explanation-neighbor-count', type=int, help='The number of items to return when querying for examples. Should be used only when the explanation method is `examples`.')
    parser.add_argument('--explanation-modality', type=str, default='MODALITY_UNSPECIFIED', help='Preset option specifying the modality of the uploaded model, which automatically configures the distance measurement and feature normalization for the underlying example index and queries. Accepted values are `IMAGE`, `TEXT` and `TABULAR`. Should be used only when the explanation method is `examples`.')
    parser.add_argument('--explanation-query', type=str, default='PRECISE', help='Preset option controlling parameters for query speed-precision trade-off. Accepted values are `PRECISE` and `FAST`. Should be used only when the explanation method is `examples`.')
    parser.add_argument('--explanation-nearest-neighbor-search-config-file', help='Path to a local JSON file that contains the configuration for the generated index,\nthe semantics are the same as metadata and should match NearestNeighborSearchConfig.\nIf you specify this parameter, no need to use `explanation-modality` and `explanation-query` for preset.\nShould be used only when the explanation method is `examples`.\n\nAn example of a JSON config file:\n\n    {\n    "contentsDeltaUri": "",\n    "config": {\n        "dimensions": 50,\n        "approximateNeighborsCount": 10,\n        "distanceMeasureType": "SQUARED_L2_DISTANCE",\n        "featureNormType": "NONE",\n        "algorithmConfig": {\n            "treeAhConfig": {\n                "leafNodeEmbeddingCount": 1000,\n                "leafNodesToSearchPercent": 100\n            }\n        }\n      }\n    }\n')