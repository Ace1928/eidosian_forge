from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootHarm(_messages.Message):
    """A LearningGenaiRootHarm object.

  Fields:
    contextualDangerous: Please do not use, this is still under development.
    csam: A boolean attribute.
    fringe: A boolean attribute.
    grailImageHarmType: A LearningGenaiRootHarmGrailImageHarmType attribute.
    grailTextHarmType: A LearningGenaiRootHarmGrailTextHarmType attribute.
    imageChild: A boolean attribute.
    imageCsam: A boolean attribute.
    imagePedo: A boolean attribute.
    imagePorn: Image signals
    imageViolence: A boolean attribute.
    pqc: A boolean attribute.
    safetycat: A LearningGenaiRootHarmSafetyCatCategories attribute.
    spii: Spii Filter uses buckets
      http://google3/google/privacy/dlp/v2/storage.proto;l=77;rcl=584719820 to
      classify the input. LMRoot converts the bucket into double score. For
      example the score for "POSSIBLE" is 3 / 5 = 0.6 .
    threshold: A number attribute.
    videoFrameChild: A boolean attribute.
    videoFrameCsam: A boolean attribute.
    videoFramePedo: A boolean attribute.
    videoFramePorn: Video frame signals
    videoFrameViolence: A boolean attribute.
  """
    contextualDangerous = _messages.BooleanField(1)
    csam = _messages.BooleanField(2)
    fringe = _messages.BooleanField(3)
    grailImageHarmType = _messages.MessageField('LearningGenaiRootHarmGrailImageHarmType', 4)
    grailTextHarmType = _messages.MessageField('LearningGenaiRootHarmGrailTextHarmType', 5)
    imageChild = _messages.BooleanField(6)
    imageCsam = _messages.BooleanField(7)
    imagePedo = _messages.BooleanField(8)
    imagePorn = _messages.BooleanField(9)
    imageViolence = _messages.BooleanField(10)
    pqc = _messages.BooleanField(11)
    safetycat = _messages.MessageField('LearningGenaiRootHarmSafetyCatCategories', 12)
    spii = _messages.MessageField('LearningGenaiRootHarmSpiiFilter', 13)
    threshold = _messages.FloatField(14)
    videoFrameChild = _messages.BooleanField(15)
    videoFrameCsam = _messages.BooleanField(16)
    videoFramePedo = _messages.BooleanField(17)
    videoFramePorn = _messages.BooleanField(18)
    videoFrameViolence = _messages.BooleanField(19)