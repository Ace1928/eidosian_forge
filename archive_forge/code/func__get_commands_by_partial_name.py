import inspect
import logging
import stevedore
def _get_commands_by_partial_name(args, commands):
    n = len(args)
    candidates = []
    for command_name in commands:
        command_parts = command_name.split()
        if len(command_parts) != n:
            continue
        if all((command_parts[i].startswith(args[i]) for i in range(n))):
            candidates.append(command_name)
    return candidates