from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1CheckConsentRequest(_messages.Message):
    """Request message for check consents.

  Fields:
    agreement: Required. Agreement to be checked against. A valid format would
      be - commerceoffercatalog.googleapis.com/billingAccounts/{billing_accoun
      t}/offers/{offer_id}/agreements/{agreement_id} commerceoffercatalog.goog
      leapis.com/services/{service}/standardOffers/{offer_id}/agreements/{agre
      ement_id}
    financialContract: Financial contract this consent applies to. This is a
      system full resource name. E.g.: //commerceoffercatalog.googleapis.com/b
      illingAccounts/{billing_account}/offers/{offer-id}
    languageCode: The language code is used to find the agreement document if
      check consent doesn't pass. If this field is set, the method will
      attempt to locate the agreement document written in that language. If
      such document cannot be found, the request will fail. If this field is
      not set, the method will try two strategies in the following order: 1)
      reuse the language of document associated with the most recent consent,
      2) use the document written in the default language, while default
      language is set by the agreement owner. If neither strategy works, the
      request will fail. Please follow BCP 47
      (https://www.w3.org/International/articles/bcp47/) for the language
      string.
    offer: Offer associated with the consent. Formats include "commerceofferca
      talog.googleapis.com/billingAccounts/{billing_account}/offers/{offer_id}
      ". "commerceoffercatalog.googleapis.com/services/{service}/standardOffer
      s/{offer_id}".
  """
    agreement = _messages.StringField(1)
    financialContract = _messages.StringField(2)
    languageCode = _messages.StringField(3)
    offer = _messages.StringField(4)