from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
def GetSuccessMessageForSynchronousDeploy(service):
    """Returns a user message for a successful synchronous deploy.

  Args:
    service: googlecloudsdk.api_lib.run.service.Service, Deployed service for
      which to build a success message.
  """
    latest_ready = service.status.latestReadyRevisionName
    latest_percent_traffic = service.latest_percent_traffic
    msg = 'Service [{{bold}}{serv}{{reset}}] revision [{{bold}}{rev}{{reset}}] has been deployed and is serving {{bold}}{latest_percent_traffic}{{reset}} percent of traffic.'
    if latest_percent_traffic:
        msg += '\nService URL: {{bold}}{url}{{reset}}'
    latest_url = service.latest_url
    tag_url_message = ''
    if latest_url:
        tag_url_message = '\nThe revision can be reached directly at {}'.format(latest_url)
    return msg.format(serv=service.name, rev=latest_ready, url=service.domain, latest_percent_traffic=latest_percent_traffic) + tag_url_message