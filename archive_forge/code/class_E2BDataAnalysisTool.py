from __future__ import annotations
import ast
import json
import os
from io import StringIO
from sys import version_info
from typing import IO, TYPE_CHECKING, Any, Callable, List, Optional, Type, Union
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr
from langchain_community.tools import BaseTool, Tool
from langchain_community.tools.e2b_data_analysis.unparse import Unparser
class E2BDataAnalysisTool(BaseTool):
    """Tool for running python code in a sandboxed environment for data analysis."""
    name = 'e2b_data_analysis'
    args_schema: Type[BaseModel] = E2BDataAnalysisToolArguments
    session: Any
    description: str
    _uploaded_files: List[UploadedFile] = PrivateAttr(default_factory=list)

    def __init__(self, api_key: Optional[str]=None, cwd: Optional[str]=None, env_vars: Optional[EnvVars]=None, on_stdout: Optional[Callable[[str], Any]]=None, on_stderr: Optional[Callable[[str], Any]]=None, on_artifact: Optional[Callable[[Artifact], Any]]=None, on_exit: Optional[Callable[[int], Any]]=None, **kwargs: Any):
        try:
            from e2b import DataAnalysis
        except ImportError as e:
            raise ImportError('Unable to import e2b, please install with `pip install e2b`.') from e
        super().__init__(description=base_description, **kwargs)
        self.session = DataAnalysis(api_key=api_key, cwd=cwd, env_vars=env_vars, on_stdout=on_stdout, on_stderr=on_stderr, on_exit=on_exit, on_artifact=on_artifact)

    def close(self) -> None:
        """Close the cloud sandbox."""
        self._uploaded_files = []
        self.session.close()

    @property
    def uploaded_files_description(self) -> str:
        if len(self._uploaded_files) == 0:
            return ''
        lines = ['The following files available in the sandbox:']
        for f in self._uploaded_files:
            if f.description == '':
                lines.append(f'- path: `{f.remote_path}`')
            else:
                lines.append(f'- path: `{f.remote_path}` \n description: `{f.description}`')
        return '\n'.join(lines)

    def _run(self, python_code: str, run_manager: Optional[CallbackManagerForToolRun]=None, callbacks: Optional[CallbackManager]=None) -> str:
        python_code = add_last_line_print(python_code)
        if callbacks is not None:
            on_artifact = getattr(callbacks.metadata, 'on_artifact', None)
        else:
            on_artifact = None
        stdout, stderr, artifacts = self.session.run_python(python_code, on_artifact=on_artifact)
        out = {'stdout': stdout, 'stderr': stderr, 'artifacts': list(map(lambda artifact: artifact.name, artifacts))}
        return json.dumps(out)

    async def _arun(self, python_code: str, run_manager: Optional[AsyncCallbackManagerForToolRun]=None) -> str:
        raise NotImplementedError('e2b_data_analysis does not support async')

    def run_command(self, cmd: str) -> dict:
        """Run shell command in the sandbox."""
        proc = self.session.process.start(cmd)
        output = proc.wait()
        return {'stdout': output.stdout, 'stderr': output.stderr, 'exit_code': output.exit_code}

    def install_python_packages(self, package_names: Union[str, List[str]]) -> None:
        """Install python packages in the sandbox."""
        self.session.install_python_packages(package_names)

    def install_system_packages(self, package_names: Union[str, List[str]]) -> None:
        """Install system packages (via apt) in the sandbox."""
        self.session.install_system_packages(package_names)

    def download_file(self, remote_path: str) -> bytes:
        """Download file from the sandbox."""
        return self.session.download_file(remote_path)

    def upload_file(self, file: IO, description: str) -> UploadedFile:
        """Upload file to the sandbox.

        The file is uploaded to the '/home/user/<filename>' path."""
        remote_path = self.session.upload_file(file)
        f = UploadedFile(name=os.path.basename(file.name), remote_path=remote_path, description=description)
        self._uploaded_files.append(f)
        self.description = self.description + '\n' + self.uploaded_files_description
        return f

    def remove_uploaded_file(self, uploaded_file: UploadedFile) -> None:
        """Remove uploaded file from the sandbox."""
        self.session.filesystem.remove(uploaded_file.remote_path)
        self._uploaded_files = [f for f in self._uploaded_files if f.remote_path != uploaded_file.remote_path]
        self.description = self.description + '\n' + self.uploaded_files_description

    def as_tool(self) -> Tool:
        return Tool.from_function(func=self._run, name=self.name, description=self.description, args_schema=self.args_schema)