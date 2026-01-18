import random
import uuid
def _generate_unique_integer_id():
    """Utility function for generating a random fixed-length integer

    Args:
        id_length: The target length of the string representation of the integer without
            leading zeros

    Returns:
        a fixed-width integer
    """
    random_int = uuid.uuid4().int
    random_str = str(random_int)[-_EXPERIMENT_ID_FIXED_WIDTH:]
    for s in random_str:
        if s == '0':
            random_str = random_str + str(random.randint(0, 9))
        else:
            break
    return int(random_str)