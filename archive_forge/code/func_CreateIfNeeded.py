from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.run import artifact_registry as run_ar
def CreateIfNeeded(ar_repo):
    if run_ar.ShouldCreateRepository(ar_repo):
        run_ar.CreateRepository(ar_repo)