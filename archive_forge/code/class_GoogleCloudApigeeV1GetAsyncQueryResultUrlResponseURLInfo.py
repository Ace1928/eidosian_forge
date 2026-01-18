from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1GetAsyncQueryResultUrlResponseURLInfo(_messages.Message):
    """A Signed URL and the relevant metadata associated with it.

  Fields:
    md5: The MD5 Hash of the JSON data
    sizeBytes: The size of the returned file in bytes
    uri: The signed URL of the JSON data. Will be of the form
      `https://storage.googleapis.com/example-bucket/cat.jpeg?X-Goog-
      Algorithm= GOOG4-RSA-SHA256&X-Goog-Credential=example%40example-
      project.iam.gserviceaccount .com%2F20181026%2Fus-
      central1%2Fstorage%2Fgoog4_request&X-Goog-Date=20181026T18 1309Z&X-Goog-
      Expires=900&X-Goog-SignedHeaders=host&X-Goog-Signature=247a2aa45f16 9edf
      4d187d54e7cc46e4731b1e6273242c4f4c39a1d2507a0e58706e25e3a85a7dbb891d62af
      a849 6def8e260c1db863d9ace85ff0a184b894b117fe46d1225c82f2aa19efd52cf21d3
      e2022b3b868dc c1aca2741951ed5bf3bb25a34f5e9316a2841e8ff4c530b22ceaa1c5ce
      09c7cbb5732631510c2058 0e61723f5594de3aea497f195456a2ff2bdd0d13bad47289d
      8611b6f9cfeef0c46c91a455b94e90a 66924f722292d21e24d31dcfb38ce0c0f353ffa5
      a9756fc2a9f2b40bc2113206a81e324fc4fd6823 a29163fa845c8ae7eca1fcf6e5bb48b
      3200983c56c5ca81fffb151cca7402beddfc4a76b13344703 2ea7abedc098d2eb14a7`
  """
    md5 = _messages.StringField(1)
    sizeBytes = _messages.IntegerField(2)
    uri = _messages.StringField(3)