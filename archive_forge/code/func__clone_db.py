import os
import subprocess
import sys
from django.db.backends.base.creation import BaseDatabaseCreation
from .client import DatabaseClient
def _clone_db(self, source_database_name, target_database_name):
    cmd_args, cmd_env = DatabaseClient.settings_to_cmd_args_env(self.connection.settings_dict, [])
    dump_cmd = ['mysqldump', *cmd_args[1:-1], '--routines', '--events', source_database_name]
    dump_env = load_env = {**os.environ, **cmd_env} if cmd_env else None
    load_cmd = cmd_args
    load_cmd[-1] = target_database_name
    with subprocess.Popen(dump_cmd, stdout=subprocess.PIPE, env=dump_env) as dump_proc:
        with subprocess.Popen(load_cmd, stdin=dump_proc.stdout, stdout=subprocess.DEVNULL, env=load_env):
            dump_proc.stdout.close()