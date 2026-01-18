import os
from subprocess import PIPE, STDOUT, Popen
from typing import Optional, Union
from urllib.parse import urlparse
from mlflow.environment_variables import MLFLOW_DOCKER_OPENJDK_VERSION
from mlflow.utils import env_manager as em
from mlflow.utils.file_utils import _copy_project
from mlflow.utils.logging_utils import eprint
from mlflow.version import VERSION
def _java_mlflow_install_step(mlflow_home):
    maven_proxy = _get_maven_proxy()
    if mlflow_home:
        return f'# Install Java mlflow-scoring from local source\nRUN cd /opt/mlflow/mlflow/java/scoring && mvn --batch-mode package -DskipTests {maven_proxy} && mkdir -p /opt/java/jars && mv /opt/mlflow/mlflow/java/scoring/target/mlflow-scoring-*-with-dependencies.jar /opt/java/jars\n'
    else:
        return f'# Install Java mlflow-scoring from Maven Central\nRUN mvn --batch-mode dependency:copy -Dartifact=org.mlflow:mlflow-scoring:{VERSION}:pom -DoutputDirectory=/opt/java {maven_proxy}\nRUN mvn --batch-mode dependency:copy -Dartifact=org.mlflow:mlflow-scoring:{VERSION}:jar -DoutputDirectory=/opt/java/jars {maven_proxy}\nRUN cp /opt/java/mlflow-scoring-{VERSION}.pom /opt/java/pom.xml\nRUN cd /opt/java && mvn --batch-mode dependency:copy-dependencies -DoutputDirectory=/opt/java/jars {maven_proxy}\n'