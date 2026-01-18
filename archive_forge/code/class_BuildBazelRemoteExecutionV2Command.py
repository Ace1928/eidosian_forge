from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildBazelRemoteExecutionV2Command(_messages.Message):
    """A `Command` is the actual command executed by a worker running an Action
  and specifications of its environment. Except as otherwise required, the
  environment (such as which system libraries or binaries are available, and
  what filesystems are mounted where) is defined by and specific to the
  implementation of the remote execution API.

  Fields:
    arguments: The arguments to the command. The first argument specifies the
      command to run, which may be either an absolute path, a path relative to
      the working directory, or an unqualified path (without path separators)
      which will be resolved using the operating system's equivalent of the
      PATH environment variable. Path separators native to the operating
      system running on the worker SHOULD be used. If the
      `environment_variables` list contains an entry for the PATH environment
      variable, it SHOULD be respected. If not, the resolution process is
      implementation-defined. Changed in v2.3. v2.2 and older require that no
      PATH lookups are performed, and that relative paths are resolved
      relative to the input root. This behavior can, however, not be relied
      upon, as most implementations already followed the rules described
      above.
    environmentVariables: The environment variables to set when running the
      program. The worker may provide its own default environment variables;
      these defaults can be overridden using this field. Additional variables
      can also be specified. In order to ensure that equivalent Commands
      always hash to the same value, the environment variables MUST be
      lexicographically sorted by name. Sorting of strings is done by code
      point, equivalently, by the UTF-8 bytes.
    outputDirectories: A list of the output directories that the client
      expects to retrieve from the action. Only the listed directories will be
      returned (an entire directory structure will be returned as a Tree
      message digest, see OutputDirectory), as well as files listed in
      `output_files`. Other files or directories that may be created during
      command execution are discarded. The paths are relative to the working
      directory of the action execution. The paths are specified using a
      single forward slash (`/`) as a path separator, even if the execution
      platform natively uses a different separator. The path MUST NOT include
      a trailing slash, nor a leading slash, being a relative path. The
      special value of empty string is allowed, although not recommended, and
      can be used to capture the entire working directory tree, including
      inputs. In order to ensure consistent hashing of the same Action, the
      output paths MUST be sorted lexicographically by code point (or,
      equivalently, by UTF-8 bytes). An output directory cannot be duplicated
      or have the same path as any of the listed output files. An output
      directory is allowed to be a parent of another output directory.
      Directories leading up to the output directories (but not the output
      directories themselves) are created by the worker prior to execution,
      even if they are not explicitly part of the input root. DEPRECATED since
      2.1: Use `output_paths` instead.
    outputFiles: A list of the output files that the client expects to
      retrieve from the action. Only the listed files, as well as directories
      listed in `output_directories`, will be returned to the client as
      output. Other files or directories that may be created during command
      execution are discarded. The paths are relative to the working directory
      of the action execution. The paths are specified using a single forward
      slash (`/`) as a path separator, even if the execution platform natively
      uses a different separator. The path MUST NOT include a trailing slash,
      nor a leading slash, being a relative path. In order to ensure
      consistent hashing of the same Action, the output paths MUST be sorted
      lexicographically by code point (or, equivalently, by UTF-8 bytes). An
      output file cannot be duplicated, be a parent of another output file, or
      have the same path as any of the listed output directories. Directories
      leading up to the output files are created by the worker prior to
      execution, even if they are not explicitly part of the input root.
      DEPRECATED since v2.1: Use `output_paths` instead.
    outputNodeProperties: A list of keys for node properties the client
      expects to retrieve for output files and directories. Keys are either
      names of string-based NodeProperty or names of fields in NodeProperties.
      In order to ensure that equivalent `Action`s always hash to the same
      value, the node properties MUST be lexicographically sorted by name.
      Sorting of strings is done by code point, equivalently, by the UTF-8
      bytes. The interpretation of string-based properties is server-
      dependent. If a property is not recognized by the server, the server
      will return an `INVALID_ARGUMENT`.
    outputPaths: A list of the output paths that the client expects to
      retrieve from the action. Only the listed paths will be returned to the
      client as output. The type of the output (file or directory) is not
      specified, and will be determined by the server after action execution.
      If the resulting path is a file, it will be returned in an OutputFile
      typed field. If the path is a directory, the entire directory structure
      will be returned as a Tree message digest, see OutputDirectory Other
      files or directories that may be created during command execution are
      discarded. The paths are relative to the working directory of the action
      execution. The paths are specified using a single forward slash (`/`) as
      a path separator, even if the execution platform natively uses a
      different separator. The path MUST NOT include a trailing slash, nor a
      leading slash, being a relative path. In order to ensure consistent
      hashing of the same Action, the output paths MUST be deduplicated and
      sorted lexicographically by code point (or, equivalently, by UTF-8
      bytes). Directories leading up to the output paths are created by the
      worker prior to execution, even if they are not explicitly part of the
      input root. New in v2.1: this field supersedes the DEPRECATED
      `output_files` and `output_directories` fields. If `output_paths` is
      used, `output_files` and `output_directories` will be ignored!
    platform: The platform requirements for the execution environment. The
      server MAY choose to execute the action on any worker satisfying the
      requirements, so the client SHOULD ensure that running the action on any
      such worker will have the same result. A detailed lexicon for this can
      be found in the accompanying platform.md. DEPRECATED as of v2.2:
      platform properties are now specified directly in the action. See
      documentation note in the Action for migration.
    workingDirectory: The working directory, relative to the input root, for
      the command to run in. It must be a directory which exists in the input
      tree. If it is left empty, then the action is run in the input root.
  """
    arguments = _messages.StringField(1, repeated=True)
    environmentVariables = _messages.MessageField('BuildBazelRemoteExecutionV2CommandEnvironmentVariable', 2, repeated=True)
    outputDirectories = _messages.StringField(3, repeated=True)
    outputFiles = _messages.StringField(4, repeated=True)
    outputNodeProperties = _messages.StringField(5, repeated=True)
    outputPaths = _messages.StringField(6, repeated=True)
    platform = _messages.MessageField('BuildBazelRemoteExecutionV2Platform', 7)
    workingDirectory = _messages.StringField(8)