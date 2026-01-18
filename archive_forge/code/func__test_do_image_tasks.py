import argparse
from copy import deepcopy
import io
import json
import os
from unittest import mock
import sys
import tempfile
import testtools
from glanceclient.common import utils
from glanceclient import exc
from glanceclient import shell
from glanceclient.v2 import shell as test_shell  # noqa
def _test_do_image_tasks(self, verbose=False, supported=True):
    args = self._make_args({'id': 'pass', 'verbose': verbose})
    expected_columns = ['Message', 'Status', 'Updated at']
    expected_output = {'tasks': [{'image_id': 'pass', 'id': 'task_1', 'user_id': 'user_1', 'request_id': 'request_id_1', 'message': 'fake_message', 'status': 'status'}]}
    if verbose:
        columns_to_prepend = ['Image Id', 'Task Id']
        columns_to_extend = ['User Id', 'Request Id', 'Result', 'Owner', 'Input', 'Expires at']
        expected_columns = columns_to_prepend + expected_columns + columns_to_extend
        expected_output['tasks'][0]['Result'] = 'Fake Result'
        expected_output['tasks'][0]['Owner'] = 'Fake Owner'
        expected_output['tasks'][0]['Input'] = 'Fake Input'
        expected_output['tasks'][0]['Expires at'] = 'Fake Expiry'
    with mock.patch.object(self.gc.images, 'get_associated_image_tasks') as mocked_tasks:
        if supported:
            mocked_tasks.return_value = expected_output
        else:
            mocked_tasks.side_effect = exc.HTTPNotImplemented
        test_shell.do_image_tasks(self.gc, args)
        mocked_tasks.assert_called_once_with('pass')
        if supported:
            utils.print_dict_list.assert_called_once_with(expected_output['tasks'], expected_columns)