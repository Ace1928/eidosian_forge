from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SystemParameters(_messages.Message):
    """### System parameter configuration  A system parameter is a special kind
  of parameter defined by the API system, not by an individual API. It is
  typically mapped to an HTTP header and/or a URL query parameter. This
  configuration specifies which methods change the names of the system
  parameters.

  Fields:
    rules: Define system parameters.  The parameters defined here will
      override the default parameters implemented by the system. If this field
      is missing from the service config, default system parameters will be
      used. Default system parameters and names is implementation-dependent.
      Example: define api key and alt name for all methods  system_parameters
      rules:     - selector: "*"       parameters:         - name: api_key
      url_query_parameter: api_key         - name: alt           http_header:
      Response-Content-Type  Example: define 2 api key names for a specific
      method.  system_parameters   rules:     - selector: "/ListShelves"
      parameters:         - name: api_key           http_header: Api-Key1
      - name: api_key           http_header: Api-Key2
  """
    rules = _messages.MessageField('SystemParameterRule', 1, repeated=True)