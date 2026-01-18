from typing import Dict, List, Optional, Union

# Type Aliases for clarity
Element = str
CompositeSymbol = List[List[str]]
EnhancedChar = List[str]
EnhancedText = List[EnhancedChar]

# Enhanced graphical elements meticulously designed for intricate detailing and advanced ASCII art construction.
# Each element is constructed within a 3x3 character grid, allowing for complex and detailed ASCII art creations.
# The naming convention for each element follows a systematic approach, indicating the level of detail and the specific feature it represents.

enhanced_elements: Dict[str, Element] = {
    # Basic empty space, serving as the foundation for more intricate designs.
    "empty_3x3": "   \n   \n   ",
    # Dots of varying intricacy, providing the basis for texture and detail.
    "dot_simple": ".  \n   \n   ",
    "dot_centered": " . \n   \n   ",
    "dot_filled": "...",
    "dot_detailed": "•••\n • \n•••",
    "dot_exquisite": "∴∴∴\n ∴ \n∴∴∴",  # Adding more intricate dot designs
    # Dashes, evolving from simple lines to more detailed representations.
    "dash_simple": "---",
    "dash_dotted": "-.-",
    "dash_double": "═╌═",
    "dash_elaborate": "─━─",
    "dash_extravagant": "┄┅┄",  # Introducing an even more detailed dash design
    # Vertical lines, from basic to embellished.
    "vertical_line_simple": "|  \n|  \n|  ",
    "vertical_line_dotted": ".  \n.  \n.  ",
    "vertical_line_double": "║  \n║  \n║  ",
    "vertical_line_elaborate": "┃  \n┃  \n┃  ",
    "vertical_line_exquisite": "┇  \n┇  \n┇  ",  # Enhancing the vertical line with a more intricate design
    # Corners, each progressively more detailed, allowing for intricate border constructions.
    "corner_top_left_simple": "+--\n|  \n|  ",
    "corner_top_right_simple": "--+\n  |\n  |",
    "corner_bottom_left_simple": "|  \n|  \n+--",
    "corner_bottom_right_simple": "  |\n  |\n--+",
    "corner_top_left_double": "╔══\n║  \n║  ",
    "corner_top_right_double": "══╗\n  ║\n  ║",
    "corner_bottom_left_double": "║  \n║  \n╚══",
    "corner_bottom_right_double": "  ║\n  ║\n══╝",
    "corner_top_left_elaborate": "╭─╮\n│  \n│  ",
    "corner_top_right_elaborate": "╮─╭\n  │\n  │",
    "corner_bottom_left_elaborate": "│  \n│  \n╰─╯",
    "corner_bottom_right_elaborate": "  │\n  │\n╯─╰",
    "corner_top_left_exquisite": "┏━┓\n┃  \n┃  ",  # Introducing a new level of corner detailing
    "corner_top_right_exquisite": "┓━┏\n  ┃\n  ┃",
    "corner_bottom_left_exquisite": "┃  \n┃  \n┗━┛",
    "corner_bottom_right_exquisite": "  ┃\n  ┃\n┛━┗",
    # Crosses, starting from simple intersections to complex junctions.
    "cross_simple": "+  \n|  \n+  ",
    "cross_detailed": "╬  \n║  \n╬  ",
    "cross_elaborate": "╋  \n┃  \n╋  ",
    "cross_exquisite": "✢  \n✣  \n✢  ",  # Adding a more intricate cross design
    # Additional intricate elements for advanced ASCII art creations.
    "diagonal_line_descending": "\\  \n \\ \n  \\",
    "diagonal_line_ascending": "  /\n / \n/",
    "circle_small": " ◯ \n◯◯\n ◯ ",
    "star": " ★ \n★★\n ★ ",
    "heart": " ❤ \n❤❤\n ❤ ",
    "spiral": " ╲ \n╱╲\n ╱ ",
    "wave": " ~ \n~~\n ~ ",
    "diagonal_line_exquisite": "╲  \n ╲ \n  ╲",  # Enhancing the diagonal line with a more detailed design
    "circle_exquisite": " ◉ \n◉◉\n ◉ ",  # Introducing a more detailed circle design
    "star_exquisite": " ✦ \n✧✧\n ✦ ",  # Adding a more intricate star design
    "heart_exquisite": " 💖 \n💖💖\n 💖 ",  # Enhancing the heart design with more detail
    "spiral_exquisite": " 🌀 \n🌀🌀\n 🌀 ",  # Introducing a more intricate spiral design
    "wave_exquisite": " ≈ \n≈≈\n ≈ ",  # Adding a more detailed wave design
}

composite_symbols: Dict[str, CompositeSymbol] = {
    # Composite symbols made from basic elements, systematically and methodically constructed for detailed ASCII art.
    # Example composite symbol
    "simple_box": [
        ["corner_top_left_simple", "dash_simple", "corner_top_right_simple"],
        ["vertical_line_simple", "empty_3x3", "vertical_line_simple"],
        ["corner_bottom_left_simple", "dash_simple", "corner_bottom_right_simple"],
    ],
    "detailed_box": [
        ["corner_top_left_double", "dash_double", "corner_top_right_double"],
        ["vertical_line_double", "empty_3x3", "vertical_line_double"],
        ["corner_bottom_left_double", "dash_double", "corner_bottom_right_double"],
    ],
    "elaborate_box": [
        ["corner_top_left_elaborate", "dash_elaborate", "corner_top_right_elaborate"],
        ["vertical_line_elaborate", "empty_3x3", "vertical_line_elaborate"],
        [
            "corner_bottom_left_elaborate",
            "dash_elaborate",
            "corner_bottom_right_elaborate",
        ],
    ],
    "exquisite_box": [
        ["corner_top_left_exquisite", "dash_extravagant", "corner_top_right_exquisite"],
        [
            "corner_bottom_left_exquisite",
            "dash_extravagant",
            "corner_bottom_right_exquisite",
        ],
    ],
    "simple_frame": [
        [
            "corner_top_left_simple",
            "dash_simple",
            "dash_simple",
            "corner_top_right_simple",
        ],
        ["vertical_line_simple", "empty_3x3", "empty_3x3", "vertical_line_simple"],
        ["vertical_line_simple", "empty_3x3", "empty_3x3", "vertical_line_simple"],
        [
            "corner_bottom_left_simple",
            "dash_simple",
            "dash_simple",
            "corner_bottom_right_simple",
        ],
    ],
    "detailed_frame": [
        [
            "corner_top_left_double",
            "dash_double",
            "dash_double",
            "corner_top_right_double",
        ],
        ["vertical_line_double", "empty_3x3", "empty_3x3", "vertical_line_double"],
        ["vertical_line_double", "empty_3x3", "empty_3x3", "vertical_line_double"],
        [
            "corner_bottom_left_double",
            "dash_double",
            "dash_double",
            "corner_bottom_right_double",
        ],
    ],
    "elaborate_frame": [
        [
            "corner_top_left_elaborate",
            "dash_elaborate",
            "dash_elaborate",
            "corner_top_right_elaborate",
        ],
        [
            "vertical_line_elaborate",
            "empty_3x3",
            "empty_3x3",
            "vertical_line_elaborate",
        ],
        [
            "vertical_line_elaborate",
            "empty_3x3",
            "empty_3x3",
            "vertical_line_elaborate",
        ],
        [
            "corner_bottom_left_elaborate",
            "dash_elaborate",
            "dash_elaborate",
            "corner_bottom_right_elaborate",
        ],
    ],
    "exquisite_frame": [
        [
            "corner_top_left_exquisite",
            "dash_extravagant",
            "dash_extravagant",
            "corner_top_right_exquisite",
        ],
        [
            "vertical_line_exquisite",
            "empty_3x3",
            "empty_3x3",
            "vertical_line_exquisite",
        ],
        [
            "vertical_line_exquisite",
            "empty_3x3",
            "empty_3x3",
            "vertical_line_exquisite",
        ],
        [
            "corner_bottom_left_exquisite",
            "dash_extravagant",
            "dash_extravagant",
            "corner_bottom_right_exquisite",
        ],
    ],
    "fancy_box": [
        ["corner_top_left_ornate", "dash_extravagant", "corner_top_right_ornate"],
        ["vertical_line_exquisite", "empty_3x3", "vertical_line_exquisite"],
        ["corner_bottom_left_ornate", "dash_extravagant", "corner_bottom_right_ornate"],
    ],
    "intricate_box": [
        [
            "corner_top_left_exquisite",
            "dash_extravagant",
            "dash_extravagant",
            "corner_top_right_exquisite",
        ],
        [
            "vertical_line_exquisite",
            "dot_exquisite",
            "dot_exquisite",
            "vertical_line_exquisite",
        ],
        [
            "vertical_line_exquisite",
            "dot_exquisite",
            "dot_exquisite",
            "vertical_line_exquisite",
        ],
        [
            "corner_bottom_left_exquisite",
            "dash_extravagant",
            "dash_extravagant",
            "corner_bottom_right_exquisite",
        ],
    ],
    "elegant_frame": [
        [
            "corner_top_left_exquisite",
            "wave_exquisite",
            "wave_exquisite",
            "corner_top_right_exquisite",
        ],
        [
            "vertical_line_exquisite",
            "heart_exquisite",
            "heart_exquisite",
            "vertical_line_exquisite",
        ],
        [
            "vertical_line_exquisite",
            "star_exquisite",
            "star_exquisite",
            "vertical_line_exquisite",
        ],
        [
            "corner_bottom_left_exquisite",
            "wave_exquisite",
            "wave_exquisite",
            "corner_bottom_right_exquisite",
        ],
    ],
    "exquisite_frame": [
        [
            "corner_top_left_exquisite",
            "spiral_exquisite",
            "spiral_exquisite",
            "corner_top_right_exquisite",
        ],
        [
            "vertical_line_exquisite",
            "circle_exquisite",
            "circle_exquisite",
            "vertical_line_exquisite",
        ],
        [
            "vertical_line_exquisite",
            "star_exquisite",
            "star_exquisite",
            "vertical_line_exquisite",
        ],
        [
            "corner_bottom_left_exquisite",
            "spiral_exquisite",
            "spiral_exquisite",
            "corner_bottom_right_exquisite",
        ],
    ],
}


def split_text(text):
    """
    Splits the input text into individual characters.

    Args:
        text (str): The input text to be split.

    Returns:
        List[str]: A list of individual characters from the input text.
    """
    return list(text)


def enhance_text(text_chars):
    """
    Enhances each character in the input list by replacing it with its corresponding enhanced ASCII art representation.

    Args:
        text_chars (List[str]): A list of individual characters to be enhanced.

    Returns:
        List[List[str]]: A list of enhanced ASCII art representations for each character.
    """
    enhanced_chars = []
    for char in text_chars:
        if char.isalpha():
            enhanced_chars.append(alphabet[char.upper()])
        elif char.isdigit():
            enhanced_chars.append(digits[char])
        elif char.isspace():
            enhanced_chars.append(space)
        else:
            enhanced_chars.append(punctuation.get(char, unknown))
    return enhanced_chars


def join_enhanced_text(enhanced_chars):
    """
    Joins the enhanced ASCII art representations of characters into a single formatted string.

    Args:
        enhanced_chars (List[List[str]]): A list of enhanced ASCII art representations for each character.

    Returns:
        str: The formatted string containing the enhanced ASCII art text.
    """
    formatted_text = []
    for i in range(3):
        row = []
        for char in enhanced_chars:
            row.append(char[i])
        formatted_text.append("".join(row))
    return "\n".join(formatted_text)


def apply_composite_symbol(text, symbol_name):
    """
    Applies a composite symbol to the enhanced text.

    Args:
        text (str): The enhanced text to apply the composite symbol to.
        symbol_name (str): The name of the composite symbol to apply.

    Returns:
        str: The enhanced text with the composite symbol applied.
    """
    symbol = composite_symbols[symbol_name]
    text_lines = text.split("\n")
    formatted_text = []

    for i in range(3):
        row = []
        for element in symbol[i]:
            row.append(enhanced_elements[element])
        formatted_text.append("".join(row))

    for line in text_lines:
        formatted_text.append(
            enhanced_elements[symbol[1][0]] + line + enhanced_elements[symbol[1][2]]
        )

    for i in range(3):
        row = []
        for element in symbol[2]:
            row.append(enhanced_elements[element])
        formatted_text.append("".join(row))

    return "\n".join(formatted_text)


def reformat_text(text, composite_symbol=None):
    """
    Reformats the input text into enhanced ASCII art.

    Args:
        text (str): The input text to be reformatted.
        composite_symbol (str): The name of the composite symbol to apply to the enhanced text.

    Returns:
        str: The reformatted text in enhanced ASCII art.
    """
    text_chars = split_text(text)
    enhanced_chars = enhance_text(text_chars)
    enhanced_text = join_enhanced_text(enhanced_chars)
    if composite_symbol:
        enhanced_text = apply_composite_symbol(enhanced_text, composite_symbol)
    return enhanced_text


def reformat_file(file_path, output_path, composite_symbol=None):
    """
    Reformats the text from a file into enhanced ASCII art and saves it to an output file.

    Args:
        file_path (str): The path to the input file containing the text to be reformatted.
        output_path (str): The path to the output file where the enhanced ASCII art will be saved.
        composite_symbol (str): The name of the composite symbol to apply to the enhanced text.
    """
    with open(file_path, "r") as file:
        text = file.read()
    enhanced_text = reformat_text(text, composite_symbol)
    with open(output_path, "w") as file:
        file.write(enhanced_text)


# Example usage
input_text = "Hello, World!"
output_text = reformat_text(input_text, composite_symbol="fancy_box")
print(output_text)

# Example usage with file paths
input_file = "input.txt"
output_file = "output.txt"
reformat_file(input_file, output_file, composite_symbol="fancy_box")
