from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
from googlecloudsdk.api_lib.app import appengine_api_client
from googlecloudsdk.api_lib.app import runtime_builders
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import deploy_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _ServerSideExperimentEnabled(self):
    """Evaluates whether the build on server-side experiment is enabled for the project.

      The experiment is enabled for a project if the sha256 hash of the
      projectID mod 100 is smaller than the current experiment rollout percent.

    Returns:
      false if the experiment is not enabled for this project or the
      experiment config cannot be read due to an error
    """
    runtimes_builder_root = properties.VALUES.app.runtime_builders_root.Get(required=True)
    try:
        experiment_config = runtime_builders.Experiments.LoadFromURI(runtimes_builder_root)
        experiment_percent = experiment_config.GetExperimentPercentWithDefault(runtime_builders.Experiments.TRIGGER_BUILD_SERVER_SIDE, 0)
        project_hash = int(hashlib.sha256(properties.VALUES.core.project.Get().encode('utf-8')).hexdigest(), 16) % 100
        return project_hash < experiment_percent
    except runtime_builders.ExperimentsError as e:
        log.debug('Experiment config file could not be read. This error is informational, and does not cause a deployment to fail. Reason: %s' % e, exc_info=True)
        return False