import re
from urllib.parse import urlparse, urlunparse
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.http_artifact_repo import HttpArtifactRepository
from mlflow.tracking._tracking_service.utils import get_tracking_uri
class MlflowArtifactsRepository(HttpArtifactRepository):
    """Scheme wrapper around HttpArtifactRepository for mlflow-artifacts server functionality"""

    def __init__(self, artifact_uri):
        super().__init__(self.resolve_uri(artifact_uri, get_tracking_uri()))

    @classmethod
    def resolve_uri(cls, artifact_uri, tracking_uri):
        base_url = '/api/2.0/mlflow-artifacts/artifacts'
        track_parse = urlparse(tracking_uri)
        uri_parse = urlparse(artifact_uri)
        _validate_port_mapped_to_hostname(uri_parse)
        _validate_uri_scheme(track_parse.scheme)
        if uri_parse.path == '/':
            resolved = f'{base_url}{uri_parse.path}'
        elif uri_parse.path == base_url:
            resolved = base_url
        else:
            resolved = f'{track_parse.path}/{base_url}/{uri_parse.path}'
        resolved = re.sub('//+', '/', resolved)
        resolved_artifacts_uri = urlunparse((track_parse.scheme, uri_parse.netloc if uri_parse.netloc else track_parse.netloc, resolved, '', '', ''))
        return resolved_artifacts_uri.replace('///', '/').rstrip('/')