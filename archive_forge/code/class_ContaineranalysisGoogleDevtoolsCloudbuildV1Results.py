from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisGoogleDevtoolsCloudbuildV1Results(_messages.Message):
    """Artifacts created by the build pipeline.

  Fields:
    artifactManifest: Path to the artifact manifest for non-container
      artifacts uploaded to Cloud Storage. Only populated when artifacts are
      uploaded to Cloud Storage.
    artifactTiming: Time to push all non-container artifacts to Cloud Storage.
    buildStepImages: List of build step digests, in the order corresponding to
      build step indices.
    buildStepOutputs: List of build step outputs, produced by builder images,
      in the order corresponding to build step indices. [Cloud
      Builders](https://cloud.google.com/cloud-build/docs/cloud-builders) can
      produce this output by writing to `$BUILDER_OUTPUT/output`. Only the
      first 50KB of data is stored.
    images: Container images that were built as a part of the build.
    mavenArtifacts: Maven artifacts uploaded to Artifact Registry at the end
      of the build.
    npmPackages: Npm packages uploaded to Artifact Registry at the end of the
      build.
    numArtifacts: Number of non-container artifacts uploaded to Cloud Storage.
      Only populated when artifacts are uploaded to Cloud Storage.
    pythonPackages: Python artifacts uploaded to Artifact Registry at the end
      of the build.
  """
    artifactManifest = _messages.StringField(1)
    artifactTiming = _messages.MessageField('ContaineranalysisGoogleDevtoolsCloudbuildV1TimeSpan', 2)
    buildStepImages = _messages.StringField(3, repeated=True)
    buildStepOutputs = _messages.BytesField(4, repeated=True)
    images = _messages.MessageField('ContaineranalysisGoogleDevtoolsCloudbuildV1BuiltImage', 5, repeated=True)
    mavenArtifacts = _messages.MessageField('ContaineranalysisGoogleDevtoolsCloudbuildV1UploadedMavenArtifact', 6, repeated=True)
    npmPackages = _messages.MessageField('ContaineranalysisGoogleDevtoolsCloudbuildV1UploadedNpmPackage', 7, repeated=True)
    numArtifacts = _messages.IntegerField(8)
    pythonPackages = _messages.MessageField('ContaineranalysisGoogleDevtoolsCloudbuildV1UploadedPythonPackage', 9, repeated=True)