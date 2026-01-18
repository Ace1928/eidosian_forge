import dataclasses
from enum import auto, Enum
from typing import Collection, Dict, List, Mapping, Optional, Set, Tuple, Union
def _sig_to_specs(*, user_inputs: Set[str], inputs_to_parameters: Mapping[str, str], inputs_to_buffers: Mapping[str, str], user_outputs: Set[str], buffer_mutations: Mapping[str, str], user_input_mutations: Mapping[str, str], grad_params: Mapping[str, str], grad_user_inputs: Mapping[str, str], loss_output: Optional[str], inputs: List[ArgumentSpec], outputs: List[ArgumentSpec]) -> Tuple[List[InputSpec], List[OutputSpec]]:

    def to_input_spec(i: ArgumentSpec) -> InputSpec:
        if not isinstance(i, TensorArgument):
            return InputSpec(kind=InputKind.USER_INPUT, arg=i, target=None)
        name = i.name
        if name in user_inputs:
            return InputSpec(kind=InputKind.USER_INPUT, arg=i, target=None)
        elif name in inputs_to_parameters:
            return InputSpec(kind=InputKind.PARAMETER, arg=i, target=inputs_to_parameters[name])
        elif name in inputs_to_buffers:
            return InputSpec(kind=InputKind.BUFFER, arg=i, target=inputs_to_buffers[name])
        else:
            raise AssertionError(f'Unknown tensor input kind: {name}')

    def to_output_spec(idx: int, o: ArgumentSpec) -> OutputSpec:
        if not isinstance(o, TensorArgument):
            return OutputSpec(kind=OutputKind.USER_OUTPUT, arg=o, target=None)
        name = o.name
        if idx < len(buffer_mutations) + len(user_input_mutations):
            if name in buffer_mutations:
                return OutputSpec(kind=OutputKind.BUFFER_MUTATION, arg=o, target=buffer_mutations[name])
            elif name in user_input_mutations:
                return OutputSpec(kind=OutputKind.USER_INPUT_MUTATION, arg=o, target=user_input_mutations[name])
            else:
                raise AssertionError(f'Unknown tensor mutation kind: {name}')
        elif name in user_outputs:
            return OutputSpec(kind=OutputKind.USER_OUTPUT, arg=o, target=None)
        elif name in grad_params:
            return OutputSpec(kind=OutputKind.GRADIENT_TO_PARAMETER, arg=o, target=grad_params[name])
        elif name in grad_user_inputs:
            return OutputSpec(kind=OutputKind.GRADIENT_TO_USER_INPUT, arg=o, target=grad_user_inputs[name])
        elif name == loss_output:
            return OutputSpec(kind=OutputKind.LOSS_OUTPUT, arg=o, target=None)
        else:
            raise AssertionError(f'Unknown tensor output kind: {name}')
    input_specs = [to_input_spec(i) for i in inputs]
    output_specs = [to_output_spec(idx, o) for idx, o in enumerate(outputs)]
    return (input_specs, output_specs)