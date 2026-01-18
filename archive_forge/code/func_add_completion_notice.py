import argparse
import sys
def add_completion_notice(parser):
    """Add completion argument to parser."""
    parser.add_argument('--print-completion', choices=['bash', 'zsh', 'tcsh'], action=_MissingCompletionAction, help='print shell completion script')
    return parser