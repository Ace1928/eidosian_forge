def main():
    # Example usage
    with open("example_script.py", "r") as file:
        content = file.read()

    parser = ScriptParser(content)
    imports = parser.parse_imports()
    docs = parser.parse_documentation()
    classes = parser.parse_classes()
    functions = parser.parse_functions()
    main_exec = parser.parse_main_executable()

    file_manager = FileManager()
    file_manager.create_directory("output")
    file_manager.organize_script_components(
        {"imports": imports, "docs": docs}, "output"
    )

    pseudocode = PseudocodeGenerator().generate_pseudocode([content])
    file_manager.create_file("output/pseudocode.txt", pseudocode)

    logger = Logger()
    logger.log("Script processing completed successfully", "info")


if __name__ == "__main__":
    main()
