from sentry_sdk.utils import event_from_exception, parse_version
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk._types import TYPE_CHECKING
def _data_from_document(document):
    try:
        operation_ast = get_operation_ast(document)
        data = {'query': print_ast(document)}
        if operation_ast is not None:
            data['variables'] = operation_ast.variable_definitions
            if operation_ast.name is not None:
                data['operationName'] = operation_ast.name.value
        return data
    except (AttributeError, TypeError):
        return dict()