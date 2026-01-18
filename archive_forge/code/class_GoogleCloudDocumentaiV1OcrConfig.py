from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1OcrConfig(_messages.Message):
    """Config for Document OCR.

  Fields:
    advancedOcrOptions: A list of advanced OCR options to further fine-tune
      OCR behavior. Current valid values are: - `legacy_layout`: a heuristics
      layout detection algorithm, which serves as an alternative to the
      current ML-based layout detection algorithm. Customers can choose the
      best suitable layout algorithm based on their situation.
    computeStyleInfo: Turn on font identification model and return font style
      information. Deprecated, use PremiumFeatures.compute_style_info instead.
    disableCharacterBoxesDetection: Turn off character box detector in OCR
      engine. Character box detection is enabled by default in OCR 2.0 (and
      later) processors.
    enableImageQualityScores: Enables intelligent document quality scores
      after OCR. Can help with diagnosing why OCR responses are of poor
      quality for a given input. Adds additional latency comparable to regular
      OCR to the process call.
    enableNativePdfParsing: Enables special handling for PDFs with existing
      text information. Results in better text extraction quality in such PDF
      inputs.
    enableSymbol: Includes symbol level OCR information if set to true.
    hints: Hints for the OCR model.
    premiumFeatures: Configurations for premium OCR features.
  """
    advancedOcrOptions = _messages.StringField(1, repeated=True)
    computeStyleInfo = _messages.BooleanField(2)
    disableCharacterBoxesDetection = _messages.BooleanField(3)
    enableImageQualityScores = _messages.BooleanField(4)
    enableNativePdfParsing = _messages.BooleanField(5)
    enableSymbol = _messages.BooleanField(6)
    hints = _messages.MessageField('GoogleCloudDocumentaiV1OcrConfigHints', 7)
    premiumFeatures = _messages.MessageField('GoogleCloudDocumentaiV1OcrConfigPremiumFeatures', 8)