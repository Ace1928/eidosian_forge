
    async def _validate_func_signature(self, func, *args, **kwargs):
        """
        Validates the function's signature against provided arguments and types.
        This method is asynchronous to ensure compatibility with both synchronous and asynchronous functions.
        It leverages Python's introspection capabilities to validate function signatures dynamically.

        Args:
            func (Callable): The function whose signature is being validated.
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Raises:
            TypeError: If the provided arguments do not match the function's signature.

        Returns:
            None: This method does not return any value but raises an exception on failure.
        """
        # Capture the start time for performance logging
        start_time = time.perf_counter()
        logging.debug(
            f"Validating function signature for {func.__name__} at {start_time}"
        )

        try:
            # Retrieve the function's signature and bind the provided arguments
            sig = signature(func)
            logging.debug(f"Function signature: {sig}")
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            logging.debug(f"Bound arguments with defaults applied: {bound_args}")

            # Retrieve type hints and validate each argument against its expected type
            type_hints = get_type_hints(func)
            logging.debug(f"Type hints: {type_hints}")

            # Ensure thread safety with asyncio.Lock for argument validation
            async with self.validation_lock:
                for name, value in bound_args.arguments.items():
                    expected_type = type_hints.get(
                        name, None
                    )  # Set a default value of None
                    logging.debug(
                        f"Validating argument '{name}' with value '{value}' against expected type '{expected_type}'"
                    )
                    if expected_type is not None:
                        if asyncio.iscoroutinefunction(expected_type):
                            if not asyncio.iscoroutine(
                                value
                            ) and not asyncio.iscoroutinefunction(value):
                                raise TypeError(
                                    f"Argument '{name}' must be a coroutine or a coroutine function, got {type(value)}"
                                )
                        elif not isinstance(value, expected_type):
                            raise TypeError(
                                f"Argument '{name}' must be of type {expected_type}, got type {type(value)}"
                            )
        except Exception as e:
            logging.error(
                f"Error validating function signature for {func.__name__}: {e}"
            )
            raise
        finally:
            # Log the completion of the validation process
            end_time = time.perf_counter()
            logging.debug(
                f"Validation of function signature for {func.__name__} completed in {end_time - start_time:.2f}s"
            )

    async def _get_arg_position(self, func: F, arg_name: str) -> int:
        """
        Determines the position of an argument in the function's signature.
        This method is asynchronous to ensure compatibility with both synchronous and asynchronous functions.
        It leverages Python's introspection capabilities to dynamically determine argument positions.

        Args:
            func (Callable): The function being inspected.
            arg_name (str): The name of the argument whose position is sought.

        Returns:
            int: The position of the argument in the function's signature.

        Raises:
            ValueError: If the argument name is not found in the function's signature.
        """
        # Capture the start time for performance logging
        start_time = asyncio.get_event_loop().time()
        logging.debug(
            f"Getting arg position for {arg_name} in {func.__name__} at {start_time}",
            extra={"async_mode": True},
        )

        try:
            # Determine the position of the argument in the function's signature
            parameters = list(signature(func).parameters)
            if arg_name not in parameters:
                raise ValueError(
                    f"Argument '{arg_name}' not found in {func.__name__}'s signature"
                )
            result = parameters.index(arg_name)

            # Log the determined position
            logging.debug(
                f"Argument position for {arg_name} in {func.__name__}: {result}",
                extra={"async_mode": True},
            )
        except Exception as e:
            logging.error(
                f"Error getting arg position for {arg_name} in {func.__name__}: {e}",
                exc_info=True,
                extra={"async_mode": True},
            )
            raise
        finally:
            # Log the completion of the process
            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time
            logging.debug(
                f"Getting arg position for {arg_name} in {func.__name__} completed in {execution_time:.2f}s",
                extra={"async_mode": True},
            )
            # Ensure logging is performed asynchronously without blocking the event loop
            await asyncio.sleep(0)
            return result

    async def validate_arguments(self, func: F, *args, **kwargs) -> None:
        """
        Validates the arguments passed to a function against expected type hints and custom validation rules.
        Adjusts for whether the function is a bound method (instance or class method) or a regular function
        or static method, and applies argument validation accordingly.
        This method is asynchronous to ensure compatibility with both synchronous and asynchronous functions,
        leveraging asyncio for non-blocking operations and ensuring thread safety with asyncio.Lock.

        Args:
            func (Callable): The function whose arguments are to be validated.
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Raises:
            TypeError: If an argument does not match its expected type.
            ValueError: If an argument fails custom validation rules.
        """
        # Initialize performance logging
        start_time = asyncio.get_event_loop().time()
        logging.debug(
            f"Validating arguments for {func.__name__} at {start_time} with args: {args} and kwargs: {kwargs}",
            extra={"async_mode": True},
        )

        # Adjust args for bound methods (instance or class methods)
        if inspect.ismethod(func) or (
            hasattr(func, "__self__") and func.__self__ is not None
        ):
            # For bound methods, the first argument ('self' or 'cls') should not be included in validation
            args = args[1:]

        # Attempt to bind args and kwargs to the function's signature
        try:
            bound_args = inspect.signature(func).bind_partial(*args, **kwargs)
        except TypeError as e:
            logging.error(
                f"Error binding arguments for {func.__name__}: {e}",
                extra={"async_mode": True},
            )
            raise

        bound_args.apply_defaults()
        type_hints = get_type_hints(func)

        # Ensure thread safety with asyncio.Lock for argument validation
        async with self.validation_lock:
            for name, value in bound_args.arguments.items():
                expected_type = type_hints.get(name)
                if not await self.validate_type(value, expected_type):
                    raise TypeError(
                        f"Argument '{name}' must be of type '{expected_type}', got type '{type(value)}'"
                    )

                validation_rule = self.validation_rules.get(name)
                if validation_rule:
                    valid = (
                        await validation_rule(value)
                        if asyncio.iscoroutinefunction(validation_rule)
                        else validation_rule(value)
                    )
                    if not valid:
                        raise ValueError(
                            f"Validation failed for argument '{name}' with value '{value}'"
                        )

        end_time = asyncio.get_event_loop().time()
        logging.debug(
            f"Validation completed at {end_time} taking total time of {end_time - start_time} seconds",
            extra={"async_mode": True},
        )

    async def validate_type(self, value: Any, expected_type: Any) -> bool:
        """
        Recursively validates a value against an expected type, handling generics, special forms, and complex types.
        This method is meticulously designed to be exhaustive in its approach to type validation,
        ensuring compatibility with a wide range of type annotations, including generics, special forms, and complex types.
        It leverages Python's typing module to interpret and validate against the provided type hints accurately.
        Utilizes asyncio for non-blocking operations and ensures thread safety with asyncio.Lock.

        Args:
            value (Any): The value to validate.
            expected_type (Any): The expected type against which to validate the value.

        Returns:
            bool: True if the value matches the expected type, False otherwise.
        """
        start_time = asyncio.get_event_loop().time()
        # Early exit for typing.Any, indicating any type is acceptable.
        if expected_type is Any:
            logging.debug(
                "Any type encountered, validation passed.", extra={"async_mode": True}
            )
            return True

        # Handle Union types, including Optional, by validating against each type argument until one matches.
        if get_origin(expected_type) is Union:
            logging.debug(
                f"Union type encountered: {expected_type}", extra={"async_mode": True}
            )
            return any(
                await self.validate_type(value, arg) for arg in get_args(expected_type)
            )

        # Handle special forms like Any, ClassVar, etc., assuming validation passes for these.
        if isinstance(expected_type, _SpecialForm):
            logging.debug(
                f"Special form encountered: {expected_type}", extra={"async_mode": True}
            )
            return True

        # Extract the origin type and type arguments from the expected type, if applicable.
        origin_type = get_origin(expected_type)
        type_args = get_args(expected_type)

        # Ensure thread safety with asyncio.Lock for type validation
        async with self.validation_lock:
            # Handle generic types (List[int], Dict[str, Any], etc.)
            if origin_type is not None:
                if not isinstance(value, origin_type):
                    logging.debug(
                        f"Value {value} does not match the origin type {origin_type}.",
                        extra={"async_mode": True},
                    )
                    return False
                if type_args:
                    # Validate type arguments (e.g., the 'int' in List[int])
                    if issubclass(origin_type, collections.abc.Mapping):
                        key_type, val_type = type_args
                        logging.debug(
                            f"Validating Mapping with key type {key_type} and value type {val_type}.",
                            extra={"async_mode": True},
                        )
                        return all(
                            await self.validate_type(k, key_type)
                            and await self.validate_type(v, val_type)
                            for k, v in value.items()
                        )
                    elif issubclass(
                        origin_type, collections.abc.Iterable
                    ) and not issubclass(origin_type, (str, bytes, bytearray)):
                        element_type = type_args[0]
                        logging.debug(
                            f"Validating each element in Iterable against type {element_type}.",
                            extra={"async_mode": True},
                        )
                        return all(
                            [
                                await self.validate_type(elem, element_type)
                                for elem in value
                            ]
                        )
                    # Extend to handle other generic types as needed
            else:
                # Handle non-generic types directly
                if not isinstance(value, expected_type):
                    logging.debug(
                        f"Value {value} does not match the expected non-generic type {expected_type}.",
                        extra={"async_mode": True},
                    )
                    return False
                return True

        # Fallback for unsupported types
        logging.debug(
            f"Type {expected_type} not supported by the current validation logic.",
            extra={"async_mode": True},
        )
        return False
