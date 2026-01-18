from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class PrintSettings(base.Group):
    """Print snippets to add to native tools settings files.

  The snippets provide a credentials placeholder and configurations to allow
  native tools to interact with Artifact Registry repositories.

  ## EXAMPLES

  To print a snippet to add a repository to the Gradle build.gradle file for
  repository my-repo in the current project, run:

      $ {command} gradle --repository=my-repo

  To print a snippet to add to the Maven pom.xml file for repository my-repo in
  the current project, run:

      $ {command} mvn --repository=my-repo

  To print a snippet to add to the npm .npmrc file for repository my-repo in
  the current project, run:

      $ {command} npm --repository=my-repo

  To print a snippet to add to the Python .pypirc and pip.comf files for
  repository my-repo in the current project, run:

      $ {command} python --repository=my-repo
  """
    category = base.CI_CD_CATEGORY