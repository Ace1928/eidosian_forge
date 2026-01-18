import ast
import copy
import re
import flask
import jsonschema
from oslo_config import cfg
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _evaluate_requirement(self, values, assertion_values, eval_type, regex):
    """Evaluate the incoming requirement and assertion.

        Filter the incoming assertions against the requirement values. If regex
        is specified, the assertion list is filtered by checking if any of the
        requirement regexes matches. Otherwise, the list is filtered by string
        equality with any of the allowed values.

        Once the assertion values are filtered, the output is determined by the
        evaluation type:
            any_one_of: return True if there are any matches, False otherwise
            not_any_of: return True if there are no matches, False otherwise
            blacklist: return the incoming values minus any matches
            whitelist: return only the matched values

        :param values: list of allowed values, defined in the requirement
        :type values: list
        :param assertion_values: The values from the assertion to evaluate
        :type assertion_values: list/string
        :param eval_type: determine how to evaluate requirements
        :type eval_type: string
        :param regex: perform evaluation with regex
        :type regex: boolean

        :returns: list of filtered assertion values (if evaluation type is
                  'blacklist' or 'whitelist'), or boolean indicating if the
                  assertion values fulfill the requirement (if evaluation type
                  is 'any_one_of' or 'not_any_of')

        """
    if regex:
        matches = self._evaluate_values_by_regex(values, assertion_values)
    else:
        matches = set(values).intersection(set(assertion_values))
    if eval_type == self._EvalType.ANY_ONE_OF:
        return bool(matches)
    elif eval_type == self._EvalType.NOT_ANY_OF:
        return not bool(matches)
    elif eval_type == self._EvalType.BLACKLIST:
        return list(set(assertion_values).difference(set(matches)))
    elif eval_type == self._EvalType.WHITELIST:
        return list(matches)
    else:
        raise exception.UnexpectedError(_('Unexpected evaluation type "%(eval_type)s"') % {'eval_type': eval_type})