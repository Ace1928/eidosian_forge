from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class ResponsePolicyRules(base.Group):
    """Manage your Cloud DNS response policy rules.

  ## EXAMPLES

  To create a new response policy rule with local data rrsets, run:

    $ {command} myresponsepolicyrule --response-policy="myresponsepolicy"
    --dns-name="www.zone.com."
    --local-data=name=www.zone.com.,type=CNAME,ttl=21600,rrdatas=zone.com.

  To create a new response policy rule with behavior, run:

    $ {command} myresponsepolicyrule --response-policy="myresponsepolicy"
    --dns-name="www.zone.com." --behavior=bypassResponsePolicy

  To update a new response policy rule with local data rrsets, run:

    $ {command} myresponsepolicyrule --response-policy="myresponsepolicy"
    --local-data=name=www.zone.com.,type=A,ttl=21600,rrdatas=1.2.3.4

  To update a new response policy rule with behavior, run:

    $ {command} myresponsepolicyrule --response-policy="myresponsepolicy"
    --behavior=bypassResponsePolicy
  """
    pass