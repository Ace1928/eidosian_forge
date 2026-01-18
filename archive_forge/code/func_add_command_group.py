import inspect
import logging
import stevedore
def add_command_group(self, group=None):
    """Adds another group of command entrypoints"""
    if group:
        self.load_commands(group)