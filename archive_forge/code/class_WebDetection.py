from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WebDetection(_messages.Message):
    """Relevant information for the image from the Internet.

  Fields:
    bestGuessLabels: The service's best guess as to the topic of the request
      image. Inferred from similar images on the open web.
    fullMatchingImages: Fully matching images from the Internet. Can include
      resized copies of the query image.
    pagesWithMatchingImages: Web pages containing the matching images from the
      Internet.
    partialMatchingImages: Partial matching images from the Internet. Those
      images are similar enough to share some key-point features. For example
      an original image will likely have partial matching for its crops.
    visuallySimilarImages: The visually similar image results.
    webEntities: Deduced entities from similar images on the Internet.
  """
    bestGuessLabels = _messages.MessageField('WebLabel', 1, repeated=True)
    fullMatchingImages = _messages.MessageField('WebImage', 2, repeated=True)
    pagesWithMatchingImages = _messages.MessageField('WebPage', 3, repeated=True)
    partialMatchingImages = _messages.MessageField('WebImage', 4, repeated=True)
    visuallySimilarImages = _messages.MessageField('WebImage', 5, repeated=True)
    webEntities = _messages.MessageField('WebEntity', 6, repeated=True)