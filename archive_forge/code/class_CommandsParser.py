from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
from redis.exceptions import RedisError, ResponseError
from redis.utils import str_if_bytes
class CommandsParser(AbstractCommandsParser):
    """
    Parses Redis commands to get command keys.
    COMMAND output is used to determine key locations.
    Commands that do not have a predefined key location are flagged with
    'movablekeys', and these commands' keys are determined by the command
    'COMMAND GETKEYS'.
    """

    def __init__(self, redis_connection):
        self.commands = {}
        self.initialize(redis_connection)

    def initialize(self, r):
        commands = r.command()
        uppercase_commands = []
        for cmd in commands:
            if any((x.isupper() for x in cmd)):
                uppercase_commands.append(cmd)
        for cmd in uppercase_commands:
            commands[cmd.lower()] = commands.pop(cmd)
        self.commands = commands

    def get_keys(self, redis_conn, *args):
        """
        Get the keys from the passed command.

        NOTE: Due to a bug in redis<7.0, this function does not work properly
        for EVAL or EVALSHA when the `numkeys` arg is 0.
         - issue: https://github.com/redis/redis/issues/9493
         - fix: https://github.com/redis/redis/pull/9733

        So, don't use this function with EVAL or EVALSHA.
        """
        if len(args) < 2:
            return None
        cmd_name = args[0].lower()
        if cmd_name not in self.commands:
            cmd_name_split = cmd_name.split()
            cmd_name = cmd_name_split[0]
            if cmd_name in self.commands:
                args = cmd_name_split + list(args[1:])
            else:
                self.initialize(redis_conn)
                if cmd_name not in self.commands:
                    raise RedisError(f"{cmd_name.upper()} command doesn't exist in Redis commands")
        command = self.commands.get(cmd_name)
        if 'movablekeys' in command['flags']:
            keys = self._get_moveable_keys(redis_conn, *args)
        elif 'pubsub' in command['flags'] or command['name'] == 'pubsub':
            keys = self._get_pubsub_keys(*args)
        else:
            if command['step_count'] == 0 and command['first_key_pos'] == 0 and (command['last_key_pos'] == 0):
                is_subcmd = False
                if 'subcommands' in command:
                    subcmd_name = f'{cmd_name}|{args[1].lower()}'
                    for subcmd in command['subcommands']:
                        if str_if_bytes(subcmd[0]) == subcmd_name:
                            command = self.parse_subcommand(subcmd)
                            is_subcmd = True
                if not is_subcmd:
                    return None
            last_key_pos = command['last_key_pos']
            if last_key_pos < 0:
                last_key_pos = len(args) - abs(last_key_pos)
            keys_pos = list(range(command['first_key_pos'], last_key_pos + 1, command['step_count']))
            keys = [args[pos] for pos in keys_pos]
        return keys

    def _get_moveable_keys(self, redis_conn, *args):
        """
        NOTE: Due to a bug in redis<7.0, this function does not work properly
        for EVAL or EVALSHA when the `numkeys` arg is 0.
         - issue: https://github.com/redis/redis/issues/9493
         - fix: https://github.com/redis/redis/pull/9733

        So, don't use this function with EVAL or EVALSHA.
        """
        pieces = args[0].split() + list(args[1:])
        try:
            keys = redis_conn.execute_command('COMMAND GETKEYS', *pieces)
        except ResponseError as e:
            message = e.__str__()
            if 'Invalid arguments' in message or 'The command has no key arguments' in message:
                return None
            else:
                raise e
        return keys