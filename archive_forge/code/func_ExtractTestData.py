from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import json
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.scc import securitycenter_client as sc_client
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
def ExtractTestData(test_data_input):
    """Extract test data into list structure, accept both list and dict."""
    if isinstance(test_data_input, list):
        return test_data_input
    elif isinstance(test_data_input, dict):
        if 'testData' in test_data_input:
            return test_data_input['testData']
        else:
            return None
    elif not test_data_input:
        raise InvalidTestDataFileError('Error parsing test data file: no data records defined in file')