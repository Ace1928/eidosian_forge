from ... import controldir
from ...commands import Command
from ...option import Option, RegistryOption
from . import helpers, load_fastimport
class cmd_fast_import(Command):
    """Backend for fast Bazaar data importers.

    This command reads a mixed command/data stream and creates
    branches in a Bazaar repository accordingly. The preferred
    recipe is::

      bzr fast-import project.fi project.bzr

    Numerous commands are provided for generating a fast-import file
    to use as input.
    To specify standard input as the input stream, use a
    source name of '-' (instead of project.fi). If the source name
    ends in '.gz', it is assumed to be compressed in gzip format.

    project.bzr will be created if it doesn't exist. If it exists
    already, it should be empty or be an existing Bazaar repository
    or branch. If not specified, the current directory is assumed.

    fast-import will intelligently select the format to use when
    creating a repository or branch. If you are running Bazaar 1.17
    up to Bazaar 2.0, the default format for Bazaar 2.x ("2a") is used.
    Otherwise, the current default format ("pack-0.92" for Bazaar 1.x)
    is used. If you wish to specify a custom format, use the `--format`
    option.

     .. note::

        To maintain backwards compatibility, fast-import lets you
        create the target repository or standalone branch yourself.
        It is recommended though that you let fast-import create
        these for you instead.

    :Branch mapping rules:

     Git reference names are mapped to Bazaar branch names as follows:

     * refs/heads/foo is mapped to foo
     * refs/remotes/origin/foo is mapped to foo.remote
     * refs/tags/foo is mapped to foo.tag
     * */master is mapped to trunk, trunk.remote, etc.
     * */trunk is mapped to git-trunk, git-trunk.remote, etc.

    :Branch creation rules:

     When a shared repository is created or found at the destination,
     branches are created inside it. In the simple case of a single
     branch (refs/heads/master) inside the input file, the branch is
     project.bzr/trunk.

     When a standalone branch is found at the destination, the trunk
     is imported there and warnings are output about any other branches
     found in the input file.

     When a branch in a shared repository is found at the destination,
     that branch is made the trunk and other branches, if any, are
     created in sister directories.

    :Working tree updates:

     The working tree is generated for the trunk branch. If multiple
     branches are created, a message is output on completion explaining
     how to create the working trees for other branches.

    :Custom exporters:

     The fast-export-from-xxx commands typically call more advanced
     xxx-fast-export scripts. You are welcome to use the advanced
     scripts if you prefer.

     If you wish to write a custom exporter for your project, see
     http://bazaar-vcs.org/BzrFastImport for the detailed protocol
     specification. In many cases, exporters can be written quite
     quickly using whatever scripting/programming language you like.

    :User mapping:

     Some source repositories store just the user name while Bazaar
     prefers a full email address. You can adjust user-ids while
     importing by using the --user-map option. The argument is a
     text file with lines in the format::

       old-id = new-id

     Blank lines and lines beginning with # are ignored.
     If old-id has the special value '@', then users without an
     email address will get one created by using the matching new-id
     as the domain, unless a more explicit address is given for them.
     For example, given the user-map of::

       @ = example.com
       bill = William Jones <bill@example.com>

     then user-ids are mapped as follows::

      maria => maria <maria@example.com>
      bill => William Jones <bill@example.com>

     .. note::

        User mapping is supported by both the fast-import and
        fast-import-filter commands.

    :Blob tracking:

     As some exporters (like git-fast-export) reuse blob data across
     commits, fast-import makes two passes over the input file by
     default. In the first pass, it collects data about what blobs are
     used when, along with some other statistics (e.g. total number of
     commits). In the second pass, it generates the repository and
     branches.

     .. note::

        The initial pass isn't done if the --info option is used
        to explicitly pass in information about the input stream.
        It also isn't done if the source is standard input. In the
        latter case, memory consumption may be higher than otherwise
        because some blobs may be kept in memory longer than necessary.

    :Restarting an import:

     At checkpoints and on completion, the commit-id -> revision-id
     map is saved to a file called 'fastimport-id-map' in the control
     directory for the repository (e.g. .bzr/repository). If the import
     is interrupted or unexpectedly crashes, it can be started again
     and this file will be used to skip over already loaded revisions.
     As long as subsequent exports from the original source begin
     with exactly the same revisions, you can use this feature to
     maintain a mirror of a repository managed by a foreign tool.
     If and when Bazaar is used to manage the repository, this file
     can be safely deleted.

    :Examples:

     Import a Subversion repository into Bazaar::

       svn-fast-export /svn/repo/path > project.fi
       bzr fast-import project.fi project.bzr

     Import a CVS repository into Bazaar::

       cvs2git /cvs/repo/path > project.fi
       bzr fast-import project.fi project.bzr

     Import a Git repository into Bazaar::

       cd /git/repo/path
       git fast-export --all > project.fi
       bzr fast-import project.fi project.bzr

     Import a Mercurial repository into Bazaar::

       cd /hg/repo/path
       hg fast-export > project.fi
       bzr fast-import project.fi project.bzr

     Import a Darcs repository into Bazaar::

       cd /darcs/repo/path
       darcs-fast-export > project.fi
       bzr fast-import project.fi project.bzr
    """
    hidden = False
    _see_also = ['fast-export', 'fast-import-filter', 'fast-import-info']
    takes_args = ['source', 'destination?']
    takes_options = ['verbose', Option('user-map', type=str, help='Path to file containing a map of user-ids.'), Option('info', type=str, help='Path to file containing caching hints.'), Option('trees', help="Update all working trees, not just trunk's."), Option('count', type=int, help='Import this many revisions then exit.'), Option('checkpoint', type=int, help='Checkpoint automatically every N revisions. The default is 10000.'), Option('autopack', type=int, help='Pack every N checkpoints. The default is 4.'), Option('inv-cache', type=int, help='Number of inventories to cache.'), RegistryOption.from_kwargs('mode', 'The import algorithm to use.', title='Import Algorithm', default='Use the preferred algorithm (inventory deltas).', experimental='Enable experimental features.', value_switches=True, enum_switch=False), Option('import-marks', type=str, help='Import marks from file.'), Option('export-marks', type=str, help='Export marks to file.'), RegistryOption('format', help='Specify a format for the created repository. See "bzr help formats" for details.', lazy_registry=('breezy.controldir', 'format_registry'), converter=lambda name: controldir.format_registry.make_controldir(name), value_switches=False, title='Repository format')]

    def run(self, source, destination='.', verbose=False, info=None, trees=False, count=-1, checkpoint=10000, autopack=4, inv_cache=-1, mode=None, import_marks=None, export_marks=None, format=None, user_map=None):
        load_fastimport()
        from .helpers import open_destination_directory
        from .processors import generic_processor
        control = open_destination_directory(destination, format=format)
        if info is None and source != '-':
            info = self._generate_info(source)
        if mode is None:
            mode = 'default'
        params = {'info': info, 'trees': trees, 'count': count, 'checkpoint': checkpoint, 'autopack': autopack, 'inv-cache': inv_cache, 'mode': mode, 'import-marks': import_marks, 'export-marks': export_marks}
        return _run(source, generic_processor.GenericProcessor, bzrdir=control, params=params, verbose=verbose, user_map=user_map)

    def _generate_info(self, source):
        from io import StringIO
        from fastimport import parser
        from fastimport.errors import ParsingError
        from fastimport.processors import info_processor
        from ...errors import CommandError
        stream = _get_source_stream(source)
        output = StringIO()
        try:
            proc = info_processor.InfoProcessor(verbose=True, outf=output)
            p = parser.ImportParser(stream)
            try:
                return_code = proc.process(p.iter_commands)
            except ParsingError as e:
                raise CommandError('%d: Parse error: %s' % (e.lineno, e))
            lines = output.getvalue().splitlines()
        finally:
            output.close()
            stream.seek(0)
        return lines