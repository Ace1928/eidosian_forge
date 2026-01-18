import os
import subprocess
class BaseDatabaseClient:
    """Encapsulate backend-specific methods for opening a client shell."""
    executable_name = None

    def __init__(self, connection):
        self.connection = connection

    @classmethod
    def settings_to_cmd_args_env(cls, settings_dict, parameters):
        raise NotImplementedError('subclasses of BaseDatabaseClient must provide a settings_to_cmd_args_env() method or override a runshell().')

    def runshell(self, parameters):
        args, env = self.settings_to_cmd_args_env(self.connection.settings_dict, parameters)
        env = {**os.environ, **env} if env else None
        subprocess.run(args, env=env, check=True)