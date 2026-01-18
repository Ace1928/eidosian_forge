from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import threading
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage import gcs_iam_util
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
Initializes CopyManagedFolderTask. Parent class documents arguments.