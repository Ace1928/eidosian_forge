import functools
import kimpy
from .exceptions import KIMModelNotFound, KIMModelInitializationError, KimpyError
class ComputeArguments:
    """
    Creates a KIM API ComputeArguments object from a KIM Portable Model object and
    configures it for ASE.  A ComputeArguments object is associated with a KIM Portable
    Model and is used to inform the KIM API of what the model can compute.  It is also
    used to register the data arrays that allow the KIM API to pass the atomic
    coordinates to the model and retrieve the corresponding energy and forces, etc.
    """

    def __init__(self, kim_model_wrapped, debug):
        self.kim_model_wrapped = kim_model_wrapped
        self.debug = debug
        self.compute_args = check_call(self.kim_model_wrapped.kim_model.compute_arguments_create)
        kimpy_arg_name = kimpy.compute_argument_name
        num_arguments = kimpy_arg_name.get_number_of_compute_argument_names()
        if self.debug:
            print('Number of compute_args: {}'.format(num_arguments))
        for i in range(num_arguments):
            name = check_call(kimpy_arg_name.get_compute_argument_name, i)
            dtype = check_call(kimpy_arg_name.get_compute_argument_data_type, name)
            arg_support = self.get_argument_support_status(name)
            if self.debug:
                print('Compute Argument name {:21} is of type {:7} and has support status {}'.format(*[str(x) for x in [name, dtype, arg_support]]))
            if arg_support == kimpy.support_status.required:
                if name != kimpy.compute_argument_name.partialEnergy and name != kimpy.compute_argument_name.partialForces:
                    raise KIMModelInitializationError('Unsupported required ComputeArgument {}'.format(name))
        callback_name = kimpy.compute_callback_name
        num_callbacks = callback_name.get_number_of_compute_callback_names()
        if self.debug:
            print()
            print('Number of callbacks: {}'.format(num_callbacks))
        for i in range(num_callbacks):
            name = check_call(callback_name.get_compute_callback_name, i)
            support_status = self.get_callback_support_status(name)
            if self.debug:
                print('Compute callback {:17} has support status {}'.format(str(name), support_status))
            if support_status == kimpy.support_status.required:
                raise KIMModelInitializationError('Unsupported required ComputeCallback: {}'.format(name))

    @check_call_wrapper
    def set_argument_pointer(self, compute_arg_name, data_object):
        return self.compute_args.set_argument_pointer(compute_arg_name, data_object)

    @check_call_wrapper
    def get_argument_support_status(self, name):
        return self.compute_args.get_argument_support_status(name)

    @check_call_wrapper
    def get_callback_support_status(self, name):
        return self.compute_args.get_callback_support_status(name)

    @check_call_wrapper
    def set_callback(self, compute_callback_name, callback_function, data_object):
        return self.compute_args.set_callback(compute_callback_name, callback_function, data_object)

    @check_call_wrapper
    def set_callback_pointer(self, compute_callback_name, callback, data_object):
        return self.compute_args.set_callback_pointer(compute_callback_name, callback, data_object)

    def update(self, num_particles, species_code, particle_contributing, coords, energy, forces):
        """Register model input and output in the kim_model object."""
        compute_arg_name = kimpy.compute_argument_name
        set_argument_pointer = self.set_argument_pointer
        set_argument_pointer(compute_arg_name.numberOfParticles, num_particles)
        set_argument_pointer(compute_arg_name.particleSpeciesCodes, species_code)
        set_argument_pointer(compute_arg_name.particleContributing, particle_contributing)
        set_argument_pointer(compute_arg_name.coordinates, coords)
        set_argument_pointer(compute_arg_name.partialEnergy, energy)
        set_argument_pointer(compute_arg_name.partialForces, forces)
        if self.debug:
            print('Debug: called update_kim')
            print()