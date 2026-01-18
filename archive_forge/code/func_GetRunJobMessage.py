from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
def GetRunJobMessage(release_track, job_name, repeat=False):
    """Returns a user message for how to run a job."""
    return '\nTo execute this job{repeat}, use:\ngcloud{release_track} run jobs execute {job_name}'.format(repeat=' again' if repeat else '', release_track=' {}'.format(release_track.prefix) if release_track.prefix is not None else '', job_name=job_name)