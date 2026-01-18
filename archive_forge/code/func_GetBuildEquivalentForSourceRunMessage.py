from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
def GetBuildEquivalentForSourceRunMessage(name, pack, source, subgroup=''):
    """Returns a user message for equivalent gcloud commands for source deploy.

  Args:
    name: name of the source target, which is either a service, a job or a
      worker
    pack: the pack arguments used to build the container image
    source: the location of the source
    subgroup: subgroup name for this command. Either 'jobs ', 'workers ' or
      empty for services
  """
    build_flag = ''
    if pack:
        build_flag = '--pack image=[IMAGE]'
    else:
        build_flag = '--tag [IMAGE]'
    msg = 'This command is equivalent to running `gcloud builds submit {build_flag} {source}` and `gcloud run {subgroup}deploy {name} --image [IMAGE]`\n'
    return msg.format(name=name, build_flag=build_flag, source=source, subgroup=subgroup)