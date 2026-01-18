from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.ai.custom_jobs import local_util
from googlecloudsdk.command_lib.ai.docker import build as docker_build
from googlecloudsdk.command_lib.ai.docker import utils as docker_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def UpdateWorkerPoolSpecsIfLocalPackageRequired(worker_pool_specs, job_name, project):
    """Update the given worker pool specifications if any contains local packages.

  If any given worker pool spec is specified a local package, this builds
  a Docker image from the local package and update the spec to use it.

  Args:
    worker_pool_specs: list of dict representing the arg value specified via the
      `--worker-pool-spec` flag.
    job_name: str, the display name of the custom job corresponding to the
      worker pool specs.
    project: str, id of the project to which the custom job is submitted.

  Yields:
    All updated worker pool specifications that uses the already built
    packages and are expectedly passed to a custom-jobs create RPC request.
  """
    image_built_for_first_worker = None
    if worker_pool_specs and 'local-package-path' in worker_pool_specs[0]:
        base_image = worker_pool_specs[0].pop('executor-image-uri')
        local_package = worker_pool_specs[0].pop('local-package-path')
        python_module = worker_pool_specs[0].pop('python-module', None)
        if python_module:
            script = local_util.ModuleToPath(python_module)
        else:
            script = worker_pool_specs[0].pop('script')
        output_image = worker_pool_specs[0].pop('output-image-uri', None)
        image_built_for_first_worker = _PrepareTrainingImage(project=project, job_name=job_name, base_image=base_image, local_package=local_package, script=script, output_image_name=output_image, python_module=python_module, requirements=worker_pool_specs[0].pop('requirements', None), extra_packages=worker_pool_specs[0].pop('extra-packages', None), extra_dirs=worker_pool_specs[0].pop('extra-dirs', None))
    for spec in worker_pool_specs:
        if image_built_for_first_worker and spec:
            new_spec = spec.copy()
            new_spec['container-image-uri'] = image_built_for_first_worker
            yield new_spec
        else:
            yield spec