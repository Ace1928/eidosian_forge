from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.util.apis import arg_utils
def AppendRemoteRepoConfigToRequest(messages, repo_args, request):
    """Adds remote repository config to CreateRepositoryRequest or UpdateRepositoryRequest."""
    remote_cfg = messages.RemoteRepositoryConfig()
    remote_cfg.description = repo_args.remote_repo_config_desc
    username = repo_args.remote_username
    secret = repo_args.remote_password_secret_version
    if username or secret:
        creds = messages.UpstreamCredentials()
        creds.usernamePasswordCredentials = messages.UsernamePasswordCredentials()
        if username:
            creds.usernamePasswordCredentials.username = username
        if secret:
            creds.usernamePasswordCredentials.passwordSecretVersion = secret
        remote_cfg.upstreamCredentials = creds
    if repo_args.disable_remote_validation:
        remote_cfg.disableUpstreamValidation = True
    if repo_args.remote_mvn_repo:
        remote_cfg.mavenRepository = messages.MavenRepository()
        facade, remote_input = ('Maven', repo_args.remote_mvn_repo)
        enum_message = _ChoiceToRemoteEnum(facade, remote_input)
        if enum_message:
            remote_cfg.mavenRepository.publicRepository = enum_message
        elif _IsRemoteURI(remote_input):
            remote_cfg.mavenRepository.customRepository = messages.GoogleDevtoolsArtifactregistryV1RemoteRepositoryConfigMavenRepositoryCustomRepository()
            remote_cfg.mavenRepository.customRepository.uri = remote_input
        elif _IsARRemote(remote_input):
            remote_cfg.mavenRepository.artifactRegistryRepository = messages.GoogleDevtoolsArtifactregistryV1RemoteRepositoryConfigMavenRepositoryArtifactRegistryRepository()
            remote_cfg.mavenRepository.artifactRegistryRepository.repository = remote_input
        else:
            _RaiseRemoteRepoUpstreamError(facade, remote_input)
    elif repo_args.remote_docker_repo:
        remote_cfg.dockerRepository = messages.DockerRepository()
        facade, remote_input = ('Docker', repo_args.remote_docker_repo)
        enum_message = _ChoiceToRemoteEnum(facade, remote_input)
        if enum_message:
            remote_cfg.dockerRepository.publicRepository = enum_message
        elif _IsRemoteURI(remote_input):
            remote_cfg.dockerRepository.customRepository = messages.GoogleDevtoolsArtifactregistryV1RemoteRepositoryConfigDockerRepositoryCustomRepository()
            remote_cfg.dockerRepository.customRepository.uri = remote_input
        elif _IsARRemote(remote_input):
            remote_cfg.dockerRepository.artifactRegistryRepository = messages.GoogleDevtoolsArtifactregistryV1RemoteRepositoryConfigDockerRepositoryArtifactRegistryRepository()
            remote_cfg.dockerRepository.artifactRegistryRepository.repository = remote_input
        else:
            _RaiseRemoteRepoUpstreamError(facade, remote_input)
    elif repo_args.remote_npm_repo:
        remote_cfg.npmRepository = messages.NpmRepository()
        facade, remote_input = ('Npm', repo_args.remote_npm_repo)
        enum_message = _ChoiceToRemoteEnum(facade, remote_input)
        if enum_message:
            remote_cfg.npmRepository.publicRepository = enum_message
        elif _IsRemoteURI(remote_input):
            remote_cfg.npmRepository.customRepository = messages.GoogleDevtoolsArtifactregistryV1RemoteRepositoryConfigNpmRepositoryCustomRepository()
            remote_cfg.npmRepository.customRepository.uri = remote_input
        elif _IsARRemote(remote_input):
            remote_cfg.npmRepository.artifactRegistryRepository = messages.GoogleDevtoolsArtifactregistryV1RemoteRepositoryConfigNpmRepositoryArtifactRegistryRepository()
            remote_cfg.npmRepository.artifactRegistryRepository.repository = remote_input
        else:
            _RaiseRemoteRepoUpstreamError(facade, remote_input)
    elif repo_args.remote_python_repo:
        remote_cfg.pythonRepository = messages.PythonRepository()
        facade, remote_input = ('Python', repo_args.remote_python_repo)
        enum_message = _ChoiceToRemoteEnum(facade, remote_input)
        if enum_message:
            remote_cfg.pythonRepository.publicRepository = enum_message
        elif _IsRemoteURI(remote_input):
            remote_cfg.pythonRepository.customRepository = messages.GoogleDevtoolsArtifactregistryV1RemoteRepositoryConfigPythonRepositoryCustomRepository()
            remote_cfg.pythonRepository.customRepository.uri = remote_input
        elif _IsARRemote(remote_input):
            remote_cfg.pythonRepository.artifactRegistryRepository = messages.GoogleDevtoolsArtifactregistryV1RemoteRepositoryConfigPythonRepositoryArtifactRegistryRepository()
            remote_cfg.pythonRepository.artifactRegistryRepository.repository = remote_input
        else:
            _RaiseRemoteRepoUpstreamError(facade, remote_input)
    elif repo_args.remote_apt_repo:
        remote_cfg.aptRepository = messages.AptRepository()
        facade, remote_base, remote_path = ('Apt', repo_args.remote_apt_repo, repo_args.remote_apt_repo_path)
        enum_message = _ChoiceToRemoteEnum(facade, remote_base)
        if enum_message:
            remote_cfg.aptRepository.publicRepository = messages.GoogleDevtoolsArtifactregistryV1RemoteRepositoryConfigAptRepositoryPublicRepository()
            remote_cfg.aptRepository.publicRepository.repositoryBase = enum_message
            remote_cfg.aptRepository.publicRepository.repositoryPath = remote_path
        elif _IsRemoteURI(_OsPackageUri(remote_base, remote_path)):
            remote_cfg.aptRepository.customRepository = messages.GoogleDevtoolsArtifactregistryV1RemoteRepositoryConfigAptRepositoryCustomRepository()
            remote_cfg.aptRepository.customRepository.uri = _OsPackageUri(remote_base, remote_path)
        elif _IsARRemote(remote_base):
            if remote_path:
                raise ar_exceptions.InvalidInputValueError('--remote-apt-repo-path is not supported for Artifact Registry Repository upstream.')
            remote_cfg.aptRepository.artifactRegistryRepository = messages.GoogleDevtoolsArtifactregistryV1RemoteRepositoryConfigAptRepositoryArtifactRegistryRepository()
            remote_cfg.aptRepository.artifactRegistryRepository.repository = remote_base
        else:
            _RaiseRemoteRepoUpstreamError(facade, remote_base)
    elif repo_args.remote_yum_repo:
        remote_cfg.yumRepository = messages.YumRepository()
        facade, remote_base, remote_path = ('Yum', repo_args.remote_yum_repo, repo_args.remote_yum_repo_path)
        enum_message = _ChoiceToRemoteEnum(facade, remote_base)
        if enum_message:
            remote_cfg.yumRepository.publicRepository = messages.GoogleDevtoolsArtifactregistryV1RemoteRepositoryConfigYumRepositoryPublicRepository()
            remote_cfg.yumRepository.publicRepository.repositoryBase = enum_message
            remote_cfg.yumRepository.publicRepository.repositoryPath = remote_path
        elif _IsRemoteURI(_OsPackageUri(remote_base, remote_path)):
            remote_cfg.yumRepository.customRepository = messages.GoogleDevtoolsArtifactregistryV1RemoteRepositoryConfigYumRepositoryCustomRepository()
            remote_cfg.yumRepository.customRepository.uri = _OsPackageUri(remote_base, remote_path)
        elif _IsARRemote(remote_base):
            if remote_path:
                raise ar_exceptions.InvalidInputValueError('--remote-yum-repo-path is not supported for Artifact Registry Repository upstream.')
            remote_cfg.yumRepository.artifactRegistryRepository = messages.GoogleDevtoolsArtifactregistryV1RemoteRepositoryConfigYumRepositoryArtifactRegistryRepository()
            remote_cfg.yumRepository.artifactRegistryRepository.repository = remote_base
        else:
            _RaiseRemoteRepoUpstreamError(facade, remote_base)
    else:
        return request
    request.repository.remoteRepositoryConfig = remote_cfg
    return request