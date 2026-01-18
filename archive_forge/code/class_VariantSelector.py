from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VariantSelector(_messages.Message):
    """VariantSelector defines different ways to map a given resource bundle to
  a target cluster.

  Fields:
    variantNameTemplate: Required. variant_name_template is a template string
      that can refer to variables containing cluster membership metadata such
      as location, name, and labels to determine the name of the variant for a
      target cluster. The variable syntax is similar to the unix shell
      variables. For example: "echo-${CLUSTER_NAME}-${CLUSTER_LOCATION}" will
      be expanded to "echo-member1-us-central1", "echo-member2-us-west1" for
      cluster members member1 and member2 in us-central1 and us-central2
      location respectively. If one wants to deploy a specific variant, say
      "default" to all the clusters, one can just use "default" (string
      without any variables) as the variant_name_template.
  """
    variantNameTemplate = _messages.StringField(1)