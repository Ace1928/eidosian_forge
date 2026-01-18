import numpy as np
from typing import Tuple, List

"""
CRITICALLY IMPORTANT DETAILS:
# Floating Point Precision

# 32-bit floating point numbers have a precision of 23 bits.
Decimal Precision≈Binary_Precision×log10(2)
Substituting the binary precision of 23 bits for a 32 bit float:
Decimal_Precision≈23×log10(2)≈6.92
Thus, a precision of 6 bits is chosen for the fractional part of the number as the 7th bit is unreliable.

# 64-bit floating point numbers have a precision of 52 bits.
Decimal Precision≈Binary_Precision×log10(2)
Substituting the binary precision of 52 bits for a 64 bit float:
Decimal Precision≈53×log10(2)≈53×0.3010≈15.953
Thus, a precision of 15 bits is chosen for the fractional part of the number as the 16th bit is unreliable.

#Scientific Notation
# 32-bit floating point numbers have a range of approximately 10^-38 to 10^38.
Single Precision (32-bit)
Format: 1 bit for sign, 8 bits for exponent, and 23 bits for the fraction (mantissa).
Bias: 127
Exponent Range: -126 to +127 (after bias adjustment)
Precision: Approximately 7 decimal places.
In single precision, the number is represented as:
(−1)sign×1.fraction×2exponent+−127 
where the "1." before the fraction part is the implicit leading bit (normalized form). The precision of about 7 decimal places comes from the 23 bits allocated for the fraction part.

# 64-bit floating point numbers have a range of approximately 10^-308 to 10^308.
Double Precision (64-bit)
Format: 1 bit for sign, 11 bits for exponent, and 52 bits for the fraction (mantissa).
Bias: 1023
Exponent Range: -1022 to +1023 (after bias adjustment)
Precision: Approximately 16 decimal places.
In double precision, the number is represented as:
(−1)sign×1.fraction×2exponent+−1023
where the "1." before the fraction part is the implicit leading bit (normalized form). The precision of about 16 decimal places comes from the 52 bits allocated for the fraction part.

# Real World Degrees of Freedom
    - 1. X dimension # Length corresponds to the x-axis.
    - 2. Y dimension # Width corresponds to the y-axis.
    - 3. Z dimension # Height corresponds to the z-axis.
    - 4. T dimension # Time corresponds to the t-axis.
    - 5. X Rotation # Roll corresponds to the orientation in radians around the x-axis.
    - 6. Y Rotation # Pitch corresponds to the orientation in radians around the y-axis.
    - 7. Z Rotation # Yaw corresponds to the orientation in radians around the z-axis.
    - 8. T Rotation # Time corresponds to the orientation in radians around the t-axis.
    - 9. X Scale Factor # X Scale corresponds to the scaling factor along the x-axis. From 0 to -1 for contraction, 0 - 1 for expansion.
    - 10. Y Scale Factor # Y Scale corresponds to the scaling factor along the y-axis. From 0 to -1 for contraction, 0 - 1 for expansion.
    - 11. Z Scale Factor # Z Scale corresponds to the scaling factor along the z-axis. From 0 to -1 for contraction, 0 - 1 for expansion.
    - 12. T Scale Factor # T Scale corresponds to the scaling factor along the t-axis. From 0 to -1 for contraction, 0 - 1 for expansion.
    - 13. X Curvature # X Skew corresponds to the curvature along the x-axis. From 0 to -1 for concave, 0 - 1 for convex.
    - 14. Y Curvature # Y Skew corresponds to the curvature along the y-axis. From 0 to -1 for concave, 0 - 1 for convex.
    - 15. Z Curvature # Z Skew corresponds to the curvature along the z-axis. From 0 to -1 for concave, 0 - 1 for convex.
    - 16. T Curvature # T Skew corresponds to the curvature along the t-axis. From 0 to -1 for concave, 0 - 1 for convex.


# Bit Matrix Representation of Floating Point Numbers
# Arbitrary Precision - Precision Affects Matrix Size

# Channel: For all parats of the value from 1 real up to 8 (octonion) imaginary parts. 1 for present, 0 for absent.
# Count: The number of components that make up the matrix (number of vectors) maximum of 16, for each channel, corresponding to comprehensive real world degrees of freedom. 
# Sign: 0 for negative, 1 for positive. 1 for each channel for each count.
# Figures
# Value
# Uncertainty


# Total Bits:
    Single Vector Single Dimension negative or positive:
        Value to Convert: -1.23456
        Value significant figures: 6    
        Channel Matrix: 9 bits (1 for each channel (real and 8 imaginary)) 0 for negative 1 for positive
        Count Matrix: 
        Final Representation: 
            Sig Fig Value 1: 1 Binary Value: 0001
            Sig Fig Value 2: 2 Binary Value: 0010
            Sig Fig Value 3: 3 Binary Value: 0011
            Sig Fig Value 4: 4 Binary Value: 0100
            Sig Fig Value 5: 5 Binary Value: 0101
            Sig Fig Value 6: 6 Binary Value: 0110
            Value Matrix = [(0,0,0,1),(0,0,1,0),(0,0,1,1),(0,1,0,0),(0,1,0,1),(0,1,1,0)]
        Bit Matrix Representation:
            [Channel Matrix], [Count Matrix], [Sign Matrix], [Value Matrix], [Uncertainty Matrix]
            Bit Matrix: (TypeForPython):  = ([1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0], [(0,0,0,1),(0,0,1,0),(0,0,1,1),(0,1,0,0),(0,1,0,1),(0,1,1,0)], [0])
    Single Vector 2 Dimensions: 
        Value to Convert: Real: -1.23456, Imaginary: 2.34567
        
        
        
        
The format of the bit matrix represenation allows for optimised efficient bitwise operations and comparisons.
The expected result being that the bit matrix representation can be used to perform operations on floating point numbers with arbitrary precision, allowing for more accurate calculations and comparisons and with greater efficiency than traditional floating point representations.
The key processes expected to make this achievable and significant are:
1. Conversion between floating point numbers and bit matrix representation.
2. Vector operations using bit matrix representation.
3. Comparison and logical operations using bit matrix representation.
4. Precision adjustment and error handling.
5. Integration with existing numerical and scientific computing libraries.
6. Potential for parallel processing and optimisation.
7. Potential for extension to multi-dimensional and multi-vector operations.
8. Potential for integration with machine learning and AI algorithms.
9. Potential for integration with quantum computing algorithms.
10. Potential for integration with blockchain and cryptographic algorithms.
11. Potential for integration with financial and economic algorithms.
12. Potential for integration with scientific and engineering algorithms.
13. Potential for integration with data analysis and visualisation algorithms.
14. Potential for integration with game development and simulation algorithms.
15. Potential for integration with educational and research algorithms.
16. Potential for integration with security and privacy algorithms.
17. Potential for integration with networking and communication algorithms.
18. Potential for integration with automation and robotics algorithms.
19. Potential for integration with healthcare and medical algorithms.
20. Potential for integration with environmental and sustainability algorithms.
21. Potential for integration with social and cultural algorithms.
22. Potential for integration with legal and governance algorithms.
23. Potential for integration with creative and artistic algorithms.
24. Potential for integration with historical and archival algorithms.
25. Potential for integration with future and speculative algorithms.
26. Potential for integration with interdisciplinary and cross-disciplinary algorithms.

The numerical form of the bit matrix representation is as follows:
[Sign Bit, Value Bits, Uncertainty Bit]
And the operations to convert between floating point numbers and bit matrix representation are as follows, numerically and algorithmically:
1. Convert Floating Point Number to Bit Matrix Representation
    Map sign to sign bit: 1 if negative, 0 if positive
    Aproximate Uncertainty based on value of least significant bit: 1 if value of least significant bit is greater than 5, 0 otherwise
    Map each digit of the value to the value bits: 0-9 (decimal) mapped to 000-1001 (binary) for each digit for each value bit
    For multi-dimensional mathematical operations:



"""

#####################################################################################
#                                                                                   #
#                             #Basic Vector Operations#                             #
#                                                                                   #
#####################################################################################


def vector_add(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Add two vectors using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.

    Returns:
    np.ndarray: The resultant vector after addition.
    """
    return np.add(v1, v2)


def vector_subtract(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Subtract two vectors using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.

    Returns:
    np.ndarray: The resultant vector after subtraction.
    """
    return np.subtract(v1, v2)


def vector_dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the dot product of two vectors using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.

    Returns:
    float: The dot product of the vectors.
    """
    return np.dot(v1, v2)


def vector_multiply(v: np.ndarray, scalar: float) -> np.ndarray:
    """
    Multiply a vector by a scalar using numpy.

    Args:
    v (np.ndarray): Vector to multiply.
    scalar (float): Scalar to multiply by.

    Returns:
    np.ndarray: The resultant vector after multiplication.
    """
    return np.multiply(v, scalar)


def vector_divide(v: np.ndarray, scalar: float) -> np.ndarray:
    """
    Divide a vector by a scalar using numpy.

    Args:
    v (np.ndarray): Vector to divide.
    scalar (float): Scalar to divide by.

    Returns:
    np.ndarray: The resultant vector after division.
    """
    return np.divide(v, scalar)


#####################################################################################
#                                                                                   #
#                             #Complex Vector Operations#                           #
#                                                                                   #
#####################################################################################


def vector_rotate_2d(v: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate a 2D vector by a given angle.

    Args:
        v (np.ndarray): 2D vector to rotate.
        angle (float): Rotation angle in radians.

    Returns:
        np.ndarray: Rotated vector.
    """
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return np.dot(rotation_matrix, v)


def vector_rotate_3d(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate a 3D vector around a specified axis by a given angle.

    Args:
        v (np.ndarray): 3D vector to rotate.
        axis (np.ndarray): Axis to rotate around.
        angle (float): Rotation angle in radians.

    Returns:
        np.ndarray: Rotated vector.
    """
    axis = axis / np.linalg.norm(axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    cross_product_matrix = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    rotation_matrix = (
        cos_angle * np.eye(3)
        + sin_angle * cross_product_matrix
        + (1 - cos_angle) * np.outer(axis, axis)
    )
    return np.dot(rotation_matrix, v)


def vector_magnitude(v: np.ndarray) -> float:
    """
    Calculate the magnitude of a vector using numpy.

    Args:
    v (np.ndarray): Vector to calculate the magnitude of.

    Returns:
    float: The magnitude of the vector.
    """
    return np.linalg.norm(v)


def vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the angle between two vectors using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.

    Returns:
    float: The angle between the vectors in radians.
    """
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def vector_projection(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calculate the projection of v1 onto v2 using numpy.

    Args:
    v1 (np.ndarray): Vector to project.
    v2 (np.ndarray): Vector to project onto.

    Returns:
    np.ndarray: The projection of v1 onto v2.
    """
    return np.dot(v1, v2) / np.dot(v2, v2) * v2


def vector_rejection(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calculate the rejection of v1 from v2 using numpy.

    Args:
    v1 (np.ndarray): Vector to reject.
    v2 (np.ndarray): Vector to reject from.

    Returns:
    np.ndarray: The rejection of v1 from v2.
    """
    return v1 - vector_projection(v1, v2)


def reflect_vector(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """
    Reflect a vector across a specified axis.

    Args:
        v (np.ndarray): Vector to reflect.
        axis (np.ndarray): Axis to reflect across, must be normalized.

    Returns:
        np.ndarray: Reflected vector.
    """
    return v - 2 * np.dot(v, axis) * axis


def vector_cross_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calculate the cross product of two vectors using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.

    Returns:
    np.ndarray: The cross product of the vectors.
    """
    return np.cross(v1, v2)


def vector_triple_product(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
    """
    Calculate the triple product of three vectors using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.
    v3 (np.ndarray): Third vector.

    Returns:
    float: The triple product of the vectors.
    """
    return np.dot(v1, np.cross(v2, v3))


#####################################################################################
#                                                                                   #
#                         #Comparative Vector Operations#                           #
#                                                                                   #
#####################################################################################


def vector_angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the angle between two vectors using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.

    Returns:
    float: The angle between the vectors in radians.
    """
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def vector_is_parallel(v1: np.ndarray, v2: np.ndarray) -> bool:
    """
    Check if two vectors are parallel using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.

    Returns:
    bool: True if the vectors are parallel, False otherwise.
    """
    return np.allclose(np.cross(v1, v2), np.zeros(3))


def vector_is_orthogonal(v1: np.ndarray, v2: np.ndarray) -> bool:
    """
    Check if two vectors are orthogonal using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.

    Returns:
    bool: True if the vectors are orthogonal, False otherwise.
    """
    return np.allclose(np.dot(v1, v2), 0)


def vector_is_normalized(v: np.ndarray) -> bool:
    """
    Check if a vector is normalized using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is normalized, False otherwise.
    """
    return np.allclose(np.linalg.norm(v), 1)


def vector_is_zero(v: np.ndarray) -> bool:
    """
    Check if a vector is zero using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is zero, False otherwise.
    """
    return np.allclose(v, np.zeros(v.shape))


def vector_is_equal(v1: np.ndarray, v2: np.ndarray) -> bool:
    """
    Check if two vectors are equal using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.

    Returns:
    bool: True if the vectors are equal, False otherwise.
    """
    return np.allclose(v1, v2)


def vector_is_orthonormal(v: np.ndarray) -> bool:
    """
    Check if a vector is orthonormal using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is orthonormal, False otherwise.
    """
    return np.allclose(np.dot(v, v.T), np.eye(v.shape[0]))


def vector_is_basis(v: np.ndarray) -> bool:
    """
    Check if a vector is a basis using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is a basis, False otherwise.
    """
    return np.linalg.matrix_rank(v) == v.shape[0]


def vector_is_linearly_independent(v: np.ndarray) -> bool:
    """
    Check if a vector is linearly independent using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is linearly independent, False otherwise.
    """
    return np.linalg.matrix_rank(v) == v.shape[0]


def vector_is_linearly_dependent(v: np.ndarray) -> bool:
    """
    Check if a vector is linearly dependent using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is linearly dependent, False otherwise.
    """
    return np.linalg.matrix_rank(v) < v.shape[0]


def vector_is_orthogonal_basis(v: np.ndarray) -> bool:
    """
    Check if a vector is an orthogonal basis using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is an orthogonal basis, False otherwise.
    """
    return np.allclose(np.dot(v, v.T), np.eye(v.shape[0]))


def vector_is_orthonormal_basis(v: np.ndarray) -> bool:
    """
    Check if a vector is an orthonormal basis using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is an orthonormal basis, False otherwise.
    """
    return np.allclose(np.dot(v, v.T), np.eye(v.shape[0]))


def vector_is_normalized_basis(v: np.ndarray) -> bool:
    """
    Check if a vector is a normalized basis using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is a normalized basis, False otherwise.
    """
    return np.allclose(np.linalg.norm(v, axis=0), np.ones(v.shape[1]))


def vector_is_orthogonal_matrix(v: np.ndarray) -> bool:
    """
    Check if a vector is an orthogonal matrix using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is an orthogonal matrix, False otherwise.
    """
    return np.allclose(np.dot(v, v.T), np.eye(v.shape[0]))


def vector_is_orthonormal_matrix(v: np.ndarray) -> bool:
    """
    Check if a vector is an orthonormal matrix using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is an orthonormal matrix, False otherwise.
    """
    return np.allclose(np.dot(v, v.T), np.eye(v.shape[0]))


def vector_is_normalized_matrix(v: np.ndarray) -> bool:
    """
    Check if a vector is a normalized matrix using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is a normalized matrix, False otherwise.
    """
    return np.allclose(np.linalg.norm(v, axis=0), np.ones(v.shape[1]))


def vector_is_orthogonal_basis_matrix(v: np.ndarray) -> bool:
    """
    Check if a vector is an orthogonal basis matrix using numpy.

    Args:
    v (np.ndarray): Vector to check.

    Returns:
    bool: True if the vector is an orthogonal basis matrix, False otherwise.
    """
    return np.allclose(np.dot(v, v.T), np.eye(v.shape[0]))


#####################################################################################
#                                                                                   #
#                             #Multi-Vector Operations#                             #
#                                                                                   #
#####################################################################################


def vector_n_dot_product(v1: np.ndarray, v2: np.ndarray, n: int) -> float:
    """
    Calculate the nth dot product of two vectors using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.
    n (int): The nth dot product to calculate.

    Returns:
    float: The nth dot product of the vectors.
    """
    return np.dot(v1, v2) ** n


def vector_n_cross_product(v1: np.ndarray, v2: np.ndarray, n: int) -> np.ndarray:
    """
    Calculate the nth cross product of two vectors using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.
    n (int): The nth cross product to calculate.

    Returns:
    np.ndarray: The nth cross product of the vectors.
    """
    return np.cross(v1, v2) ** n


def vector_n_triple_product(
    v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, n: int
) -> float:
    """
    Calculate the nth triple product of three vectors using numpy.

    Args:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector.
    v3 (np.ndarray): Third vector.
    n (int): The nth triple product to calculate.

    Returns:
    float: The nth triple product of the vectors.
    """
    return np.dot(v1, np.cross(v2, v3)) ** n


def vector_dot_product_between_n_vectors(*vectors: List[float]) -> float:
    """
    Calculate the dot product between n vectors using numpy.

    Args:
    *vectors (List[float]): List of vectors.

    Returns:
    float: The dot product between the vectors.

    """
    return np.dot(*vectors)


def vector_normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to have a magnitude of 1.

    Args:
    v (np.ndarray): Vector to normalize.

    Returns:
    np.ndarray: Normalized vector.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


#####################################################################################
#                                                                                   #
#                             #Vector Utilities#                                    #
#                                                                                   #
#####################################################################################


def adjust_precision_of_bit_matrix(bm: np.ndarray, new_precision: int) -> np.ndarray:
    """
    Adjust the precision of a bit matrix representation of vectors.

    Args:
        bm (np.ndarray): Bit matrix to adjust.
        new_precision (int): New precision level.

    Returns:
        np.ndarray: Adjusted bit matrix.
    """
    # Implementation would involve recalculating the bit matrix based on new precision.
    # This is a placeholder for the actual implementation.
    pass


def decompose_vector(v: np.ndarray, bases: List[np.ndarray]) -> List[np.ndarray]:
    """
    Decompose a vector into components along a set of orthogonal bases.

    Args:
        v (np.ndarray): Vector to decompose.
        bases (List[np.ndarray]): Orthogonal bases for decomposition.

    Returns:
        List[np.ndarray]: Components of the vector along the given bases.
    """
    components = []
    for base in bases:
        projection = np.dot(v, base) / np.dot(base, base) * base
        components.append(projection)
    return components


#####################################################################################
#                                                                                   #
#                #   Vector to Bit Matrix InterConversion   #                       #
#                                                                                   #
#####################################################################################


def float_to_bit_matrix(v: np.ndarray, precision: int = 6) -> np.ndarray:
    """
    Convert a floating point vector to a bit matrix based on the specified precision.
    Each number is represented by sign, value, and uncertainty bits.

    Args:
    v (np.ndarray): Vector of floats.
    precision (int): Number of bits dedicated to the fractional part.

    Returns:
    np.ndarray: The bit matrix representing the vector.
    """
    # Normalize the vector to the range -1 to 1 for sign bit implementation.
    normalized_v = v / np.max(np.abs(v))
    bit_matrix = np.array(
        [
            [1 if num < 0 else 0]  # Sign bit
            + [
                int(bit)
                for bit in f"{abs(num):0.{precision}f}".replace(".", "")[:precision]
            ]  # Value bits
            + [0]  # Uncertainty bit, currently simplified
            for num in normalized_v
        ]
    )
    return bit_matrix


def bit_matrix_to_float(bm: np.ndarray, original_max: float) -> np.ndarray:
    """
    Convert a bit matrix back to a vector of floats.

    Args:
    bm (np.ndarray): Bit matrix.
    original_max (float): Original maximum value used for normalization.

    Returns:
    np.ndarray: The resultant vector of floats.
    """

    def bits_to_float(bits: List[int]) -> float:
        # Reconstruct the floating point number from bits
        sign = -1 if bits[0] == 1 else 1
        value_str = "".join(str(bit) for bit in bits[1:-1])
        value = sign * float(f"0.{value_str}")
        return value * original_max

    return np.array([bits_to_float(bits) for bits in bm])


#####################################################################################
#                                                                                   #
#                        #   Bit Matrix Vector Operations   #                       #
#                                                                                   #
#####################################################################################


def bitwise_and_between_bit_matrices(bm1: np.ndarray, bm2: np.ndarray) -> np.ndarray:
    """
    Perform a bitwise AND operation between two bit matrices.

    Args:
        bm1 (np.ndarray): First bit matrix.
        bm2 (np.ndarray): Second bit matrix.

    Returns:
        np.ndarray: Resultant bit matrix after bitwise AND operation.
    """
    return np.bitwise_and(bm1, bm2)


def bitwise_or_between_bit_matrices(bm1: np.ndarray, bm2: np.ndarray) -> np.ndarray:
    """
    Perform a bitwise OR operation between two bit matrices.

    Args:
        bm1 (np.ndarray): First bit matrix.
        bm2 (np.ndarray): Second bit matrix.

    Returns:
        np.ndarray: Resultant bit matrix after bitwise OR operation.
    """
    return np.bitwise_or(bm1, bm2)


def invert_bit_matrix(bm: np.ndarray) -> np.ndarray:
    """
    Invert a bit matrix, switching all 0s to 1s and vice versa.

    Args:
        bm (np.ndarray): Bit matrix to invert.

    Returns:
        np.ndarray: Inverted bit matrix.
    """
    return np.logical_not(bm).astype(int)


def reduce_bit_matrix(bm: np.ndarray, factor: int) -> np.ndarray:
    """
    Reduce the size of a bit matrix by a specified factor, combining bits by averaging.

    Args:
        bm (np.ndarray): Bit matrix to reduce.
        factor (int): Reduction factor.

    Returns:
        np.ndarray: Reduced bit matrix.
    """
    # Placeholder for actual implementation
    pass
