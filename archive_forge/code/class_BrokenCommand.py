import click
import os
import sys
import traceback
class BrokenCommand(click.Command):
    """
    Rather than completely crash the CLI when a broken plugin is loaded, this
    class provides a modified help message informing the user that the plugin is
    broken and they should contact the owner.  If the user executes the plugin
    or specifies `--help` a traceback is reported showing the exception the
    plugin loader encountered.
    """

    def __init__(self, name):
        """
        Define the special help messages after instantiating a `click.Command()`.
        """
        click.Command.__init__(self, name)
        util_name = os.path.basename(sys.argv and sys.argv[0] or __file__)
        if os.environ.get('CLICK_PLUGINS_HONESTLY'):
            icon = u'💩'
        else:
            icon = u'†'
        self.help = '\nWarning: entry point could not be loaded. Contact its author for help.\n\n\x08\n' + traceback.format_exc()
        self.short_help = icon + ' Warning: could not load plugin. See `%s %s --help`.' % (util_name, self.name)

    def invoke(self, ctx):
        """
        Print the traceback instead of doing nothing.
        """
        click.echo(self.help, color=ctx.color)
        ctx.exit(1)

    def parse_args(self, ctx, args):
        return args