from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.core.util import files
def ParseJsonFileToDeployment(deployment_file):
    f = files.ReadFileContents(deployment_file)
    return json.loads(f)