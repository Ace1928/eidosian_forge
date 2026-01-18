import argparse
import contextlib
import io
import json
import logging
import os
import pkgutil
import sys
from apitools.base.py import exceptions
from apitools.gen import gen_client_lib
from apitools.gen import util
def _WriteGeneratedFiles(args, codegen):
    if codegen.use_proto2:
        _WriteProtoFiles(codegen)
    with util.Chdir(codegen.outdir):
        with io.open(codegen.client_info.messages_file_name, 'w') as out:
            codegen.WriteMessagesFile(out)
        with io.open(codegen.client_info.client_file_name, 'w') as out:
            codegen.WriteClientLibrary(out)