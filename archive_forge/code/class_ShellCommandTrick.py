import os
import signal
import subprocess
import time
from wandb_watchdog.utils import echo, has_attribute
from wandb_watchdog.events import PatternMatchingEventHandler
class ShellCommandTrick(Trick):
    """Executes shell commands in response to matched events."""

    def __init__(self, shell_command=None, patterns=None, ignore_patterns=None, ignore_directories=False, wait_for_process=False, drop_during_process=False):
        super(ShellCommandTrick, self).__init__(patterns, ignore_patterns, ignore_directories)
        self.shell_command = shell_command
        self.wait_for_process = wait_for_process
        self.drop_during_process = drop_during_process
        self.process = None

    def on_any_event(self, event):
        from string import Template
        if self.drop_during_process and self.process and (self.process.poll() is None):
            return
        if event.is_directory:
            object_type = 'directory'
        else:
            object_type = 'file'
        context = {'watch_src_path': event.src_path, 'watch_dest_path': '', 'watch_event_type': event.event_type, 'watch_object': object_type}
        if self.shell_command is None:
            if has_attribute(event, 'dest_path'):
                context.update({'dest_path': event.dest_path})
                command = 'echo "${watch_event_type} ${watch_object} from ${watch_src_path} to ${watch_dest_path}"'
            else:
                command = 'echo "${watch_event_type} ${watch_object} ${watch_src_path}"'
        else:
            if has_attribute(event, 'dest_path'):
                context.update({'watch_dest_path': event.dest_path})
            command = self.shell_command
        command = Template(command).safe_substitute(**context)
        self.process = subprocess.Popen(command, shell=True)
        if self.wait_for_process:
            self.process.wait()