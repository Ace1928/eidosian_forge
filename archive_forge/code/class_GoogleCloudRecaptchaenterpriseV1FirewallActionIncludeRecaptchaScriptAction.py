from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1FirewallActionIncludeRecaptchaScriptAction(_messages.Message):
    """An include reCAPTCHA script action involves injecting reCAPTCHA
  JavaScript code into the HTML returned by the site backend. This reCAPTCHA
  script is tasked with collecting user signals on the requested web page,
  issuing tokens as a cookie within the site domain, and enabling their
  utilization in subsequent page requests.
  """