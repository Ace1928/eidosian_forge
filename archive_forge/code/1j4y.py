class ConfigManager:
    def load_config(self, config_path):
        """Loads configuration settings from a JSON file."""
        with open(config_path, "r") as file:
            return json.load(file)
