import multiprocessing
import os
import signal
import subprocess
import gyp.common
import gyp.xcode_emulation
def CMakeStringEscape(a):
    """Escapes the string 'a' for use inside a CMake string.

  This means escaping
  '' otherwise it may be seen as modifying the next character
  '"' otherwise it will end the string
  ';' otherwise the string becomes a list

  The following do not need to be escaped
  '#' when the lexer is in string state, this does not start a comment

  The following are yet unknown
  '$' generator variables (like ${obj}) must not be escaped,
      but text $ should be escaped
      what is wanted is to know which $ come from generator variables
  """
    return a.replace('\\', '\\\\').replace(';', '\\;').replace('"', '\\"')