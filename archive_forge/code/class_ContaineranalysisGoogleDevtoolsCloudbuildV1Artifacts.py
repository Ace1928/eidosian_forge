from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisGoogleDevtoolsCloudbuildV1Artifacts(_messages.Message):
    """Artifacts produced by a build that should be uploaded upon successful
  completion of all build steps.

  Fields:
    images: A list of images to be pushed upon the successful completion of
      all build steps. The images will be pushed using the builder service
      account's credentials. The digests of the pushed images will be stored
      in the Build resource's results field. If any of the images fail to be
      pushed, the build is marked FAILURE.
    mavenArtifacts: A list of Maven artifacts to be uploaded to Artifact
      Registry upon successful completion of all build steps. Artifacts in the
      workspace matching specified paths globs will be uploaded to the
      specified Artifact Registry repository using the builder service
      account's credentials. If any artifacts fail to be pushed, the build is
      marked FAILURE.
    npmPackages: A list of npm packages to be uploaded to Artifact Registry
      upon successful completion of all build steps. Npm packages in the
      specified paths will be uploaded to the specified Artifact Registry
      repository using the builder service account's credentials. If any
      packages fail to be pushed, the build is marked FAILURE.
    objects: A list of objects to be uploaded to Cloud Storage upon successful
      completion of all build steps. Files in the workspace matching specified
      paths globs will be uploaded to the specified Cloud Storage location
      using the builder service account's credentials. The location and
      generation of the uploaded objects will be stored in the Build
      resource's results field. If any objects fail to be pushed, the build is
      marked FAILURE.
    pythonPackages: A list of Python packages to be uploaded to Artifact
      Registry upon successful completion of all build steps. The build
      service account credentials will be used to perform the upload. If any
      objects fail to be pushed, the build is marked FAILURE.
  """
    images = _messages.StringField(1, repeated=True)
    mavenArtifacts = _messages.MessageField('ContaineranalysisGoogleDevtoolsCloudbuildV1ArtifactsMavenArtifact', 2, repeated=True)
    npmPackages = _messages.MessageField('ContaineranalysisGoogleDevtoolsCloudbuildV1ArtifactsNpmPackage', 3, repeated=True)
    objects = _messages.MessageField('ContaineranalysisGoogleDevtoolsCloudbuildV1ArtifactsArtifactObjects', 4)
    pythonPackages = _messages.MessageField('ContaineranalysisGoogleDevtoolsCloudbuildV1ArtifactsPythonPackage', 5, repeated=True)