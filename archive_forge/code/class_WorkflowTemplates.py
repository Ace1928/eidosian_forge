from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class WorkflowTemplates(base.Group):
    """Create and manage Dataproc workflow templates.

  Create and manage Dataproc workflow templates.

  ## EXAMPLES

  To create a workflow template, run:

    $ {command} create my_template

  To instantiate a workflow template, run:

    $ {command} instantiate my_template

  To instantiate a workflow template from a file, run:

    $ {command} instantiate-from-file --file template.yaml

  To delete a workflow template, run:

    $ {command} delete my_template

  To view the details of a workflow template, run:

    $ {command} describe my_template

  To see the list of all workflow templates, run:

    $ {command} list

  To remove a job from a workflow template, run:

    $ {command} remove-job my_template --step-id id

  To update managed cluster in a workflow template, run:

    $ {command} set-managed-cluster my_template --num-masters 5

  To update cluster selector in a workflow template, run:

    $ {command} set-cluster-selector my_template \\
        --cluster-labels environment=prod

  """
    pass